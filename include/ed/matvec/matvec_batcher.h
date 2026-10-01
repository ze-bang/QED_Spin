#pragma once
// =============================================================================
// include/ed/matvec/matvec_batcher.h
//
// Independent Krylov samples in lockstep: each sample runs the ordinary single-vector code on
// its own host thread, and the matvecs of all of them meet here. A call blocks until every
// running sample is waiting in one; the waiting calls then run grouped by operator, one
// multi-vector apply per operator (LinearOperator::bind_cuda_multi), and all resume. The
// kernels themselves are unchanged, and a multi-vector apply returns what single applies
// would, so a batched run reproduces the sequential one.
//
//   MatvecBatcher b;
//   auto H = b.wrap(op.bind_cuda_multi());        // wrap each operator once, share it
//   b.run(k, [&](std::size_t i) { sample(i, H); });
//
// Device work is ordered by CUDA's legacy default stream (stream 0) the backends and kernels share, so a
// launch made by one thread follows everything the others queued before they blocked.
// =============================================================================

#include <ed/core/linear_operator.h>

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ed::matvec {

class MatvecBatcher {
public:
    using Multi  = ed::LinearOperator::MultiMatvecFn;
    using Single = std::function<void(const Complex*, Complex*, std::size_t)>;

    /// A matvec on `multi` that batches with the other samples' calls. Wrap once, share.
    Single wrap(Multi multi) {
        auto m = std::make_shared<Multi>(std::move(multi));
        ops_.push_back(m);
        return [this, op = m.get()](const Complex* in, Complex* out, std::size_t n) { call(op, in, out, n); };
    }

    /// fn(i) for i in [0, k), each on its own thread; the first exception is rethrown after all
    /// threads have finished.
    template <class Fn>
    void run(std::size_t k, Fn&& fn) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            active_ = k;
            error_  = nullptr;
        }
        std::exception_ptr first;
        std::mutex first_mu;
        std::vector<std::thread> threads;
        threads.reserve(k);
        for (std::size_t i = 0; i < k; ++i)
            threads.emplace_back([&, i] {
                try {
                    fn(i);
                } catch (...) {
                    std::lock_guard<std::mutex> lk(first_mu);
                    if (!first) first = std::current_exception();
                }
                leave();
            });
        for (auto& t : threads) t.join();
        if (first) std::rethrow_exception(first);
    }

private:
    struct Call { const Multi* op; const Complex* in; Complex* out; std::size_t n; };

    void call(const Multi* op, const Complex* in, Complex* out, std::size_t n) {
        std::unique_lock<std::mutex> lk(mu_);
        if (error_) std::rethrow_exception(error_);
        pending_.push_back({op, in, out, n});
        const std::uint64_t gen = gen_;
        if (pending_.size() == active_) flush();
        else cv_.wait(lk, [&] { return gen_ != gen; });
        if (error_) std::rethrow_exception(error_);
    }

    // A sample finished: the others may now all be waiting.
    void leave() {
        std::lock_guard<std::mutex> lk(mu_);
        --active_;
        if (active_ > 0 && pending_.size() == active_) flush();
    }

    // Under the lock, with every running sample blocked in call().
    void flush() {
        std::vector<Call> calls;
        calls.swap(pending_);
        try {
            std::map<const Multi*, std::vector<const Call*>> by_op;
            for (const Call& c : calls) by_op[c.op].push_back(&c);
            for (const auto& [op, cs] : by_op) {
                std::vector<const Complex*> ins;
                std::vector<Complex*> outs;
                for (const Call* c : cs) {
                    if (c->n != cs.front()->n) throw std::invalid_argument("MatvecBatcher: sizes differ for one operator");
                    ins.push_back(c->in);
                    outs.push_back(c->out);
                }
                (*op)(ins.data(), outs.data(), cs.front()->n, ins.size());
            }
        } catch (...) {
            error_ = std::current_exception();
        }
        ++gen_;
        cv_.notify_all();
    }

    std::vector<std::shared_ptr<Multi>> ops_;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::vector<Call>       pending_;
    std::size_t             active_ = 0;
    std::uint64_t           gen_    = 0;
    std::exception_ptr      error_;
};

}  // namespace ed::matvec
