#pragma once
// =============================================================================
// include/ed/ops/operator.h
//
// Operator: a spin-1/2 operator on n_bits sites -- its term records (one-, two- and
// three-body products of S+, S-, S^z) and canonical terms on more sites -- and its action on
// the full 2^N space. Implements ``ed::LinearOperator``, so solvers consume Operator and the
// symmetry-adapted operators through one polymorphic surface.
//
//   * Construction:   Operator(n_bits, spin_l)
//   * Terms:          addOneBodyTerm / addTwoBodyTerm / addThreeBodyTerm / add_record /
//                     add_extra_term; records(), three_body_records(), extra_terms()
//   * What they mean: canonical() (invariance.h), row_program() (row_walk.h)
//   * Matvec:         apply (the row walk, or the CSR assembled from it)
//   * Properties:     isReal, dim, is_hermitian
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Sparse>
#include <ed/basis/bits.h>
#include <ed/matvec/linear_operator.h>
#include <ed/core/config.h>        // ed::env (ED_CSR_FORCE, ED_CSR_DIM_MAX)
#include <ed/core/errors.h>        // ed::Unsupported
#include <ed/matvec/reduced_csr.h>  // the full-space CSR
#include <ed/ops/invariance.h>     // canonical terms, verdicts
#include <ed/ops/row_walk.h>       // for_each_connection

using Complex = std::complex<double>;

// Operator implements ed::LinearOperator: apply(), dim() and
// is_hermitian() are virtual, so solvers
// and dispatchers consume Operator and symmetry-adapted operators through
// one interface. The virtual destructor makes deletion through a base
// pointer safe.
class Operator : public ed::LinearOperator {
public:
    // ========================================================================
    // Terms: the records, in insertion order (what the builders and the bindings add), and
    // the canonical terms on four or more sites that no record holds. canonical() is the
    // operator they describe; every matvec walks the program compiled from it.
    // ========================================================================

    /// Three-body record (op_1 site_1)(op_2 site_2)(op_3 site_3), op_1 acting first.
    struct ThreeBodyTransformData {
        std::uint8_t op_type_1{0};      ///< 0 = S+, 1 = S-, 2 = Sz
        std::uint64_t site_index_1{0};
        std::uint8_t op_type_2{0};
        std::uint64_t site_index_2{0};
        std::uint8_t op_type_3{0};
        std::uint64_t site_index_3{0};
        Complex coefficient{0.0, 0.0};
    };

    /// One- or two-body record (op site)(op_2 site_2), op_2 acting first.
    struct TransformData {
        uint8_t op_type{0};         ///< 0 = S+, 1 = S-, 2 = Sz
        uint64_t site_index{0};
        Complex coefficient{0.0, 0.0};
        uint64_t site_index_2{0};
        uint8_t op_type_2{0};
        bool is_two_body{false};
    };

private:
    /// The term records in insertion order. They change only through add_record and the typed
    /// setters, which validate every factor.
    std::vector<TransformData> transform_data_;
    std::vector<ThreeBodyTransformData> three_body_data_;
    /// Canonical terms on four or more sites, which no record can hold: part of canonical()
    /// and of every row walk, invisible to the record readers (which refuse such operators).
    std::vector<ed::ops::MaskedTerm> extra_terms_;

public:
    // ------------------------------------------------------------------
    // Records: read them, append one (validated), or use the typed setters.
    // ------------------------------------------------------------------

    /// The records in insertion order (a two-body record is O1 O2 with O2 acting first).
    [[nodiscard]] const std::vector<TransformData>& records() const noexcept { return transform_data_; }
    [[nodiscard]] const std::vector<ThreeBodyTransformData>& three_body_records() const noexcept {
        return three_body_data_;
    }

    /// Append one record. Throws std::invalid_argument for an op type outside 0 (S+), 1 (S-),
    /// 2 (Sz) or a site outside [0, n_bits).
    void add_record(const TransformData& t) {
        check_factor_(t.op_type, t.site_index);
        if (t.is_two_body) check_factor_(t.op_type_2, t.site_index_2);
        transform_data_.push_back(t);
        invalidateMatrixCaches();
    }
    void add_record(const ThreeBodyTransformData& t) {
        check_factor_(t.op_type_1, t.site_index_1);
        check_factor_(t.op_type_2, t.site_index_2);
        check_factor_(t.op_type_3, t.site_index_3);
        three_body_data_.push_back(t);
        invalidateMatrixCaches();
    }

    /// Append a canonical term on four or more sites (term.h; masks within n_bits). Terms on
    /// fewer sites go in as records (ed::ops::to_operator writes both).
    void add_extra_term(const ed::ops::MaskedTerm& t) {
        const std::uint64_t all = (n_bits_ >= 64) ? ~0ULL : ((1ULL << n_bits_) - 1ULL);
        if (((t.cond_mask | t.flip_mask | t.sign_mask) & ~all) != 0)
            throw std::invalid_argument("Operator::add_extra_term: a mask names a site outside the operator");
        extra_terms_.push_back(t);
        invalidateMatrixCaches();
    }
    [[nodiscard]] const std::vector<ed::ops::MaskedTerm>& extra_terms() const noexcept { return extra_terms_; }
    [[nodiscard]] bool has_extra_terms() const noexcept { return !extra_terms_.empty(); }

    /// The operator's canonical terms (invariance.h), built on first use and kept until the
    /// records change.
    [[nodiscard]] const ed::ops::MaskedOperator& canonical() const {
        std::lock_guard<std::mutex> lock(canonical_mutex_);
        if (!canonical_) canonical_ = std::make_shared<const ed::ops::MaskedOperator>(ed::ops::masked(*this));
        return *canonical_;
    }

    /// What the row walks read (row_walk.h): the program of the ADJOINT's canonical terms, so
    /// a row of H is the conjugated walk; built on first use and kept until the records change.
    [[nodiscard]] std::shared_ptr<const ed::ops::MaskedProgram> row_program() const {
        std::lock_guard<std::mutex> lock(row_program_mutex_);
        if (!row_program_)
            row_program_ =
                std::make_shared<const ed::ops::MaskedProgram>(ed::ops::compile_operator(canonical().dagger()));
        return row_program_;
    }

    /// Append a one-body term (op_type, site, coeff) to the canonical AoS
    /// storage and invalidate the SoA cache. ``op_type``: 0 = S+, 1 = S-,
    /// 2 = Sz.
    void addOneBodyTerm(uint8_t op_type, uint64_t site, const Complex& coeff) {
        TransformData td;
        td.op_type = op_type;
        td.site_index = site;
        td.coefficient = coeff;
        td.is_two_body = false;
        add_record(td);
    }

    /// Append a two-body term (op1*site1)(op2*site2) with coupling ``coeff``.
    void addTwoBodyTerm(uint8_t op_type_1, uint64_t site_1, uint8_t op_type_2, uint64_t site_2, const Complex& coeff) {
        TransformData td;
        td.op_type = op_type_1;
        td.site_index = site_1;
        td.op_type_2 = op_type_2;
        td.site_index_2 = site_2;
        td.coefficient = coeff;
        td.is_two_body = true;
        add_record(td);
    }

    /// Append a three-body term (op1*site1)(op2*site2)(op3*site3) with
    /// coupling ``coeff``.
    void addThreeBodyTerm(uint8_t op_type_1, uint64_t site_1, uint8_t op_type_2, uint64_t site_2, uint8_t op_type_3,
                          uint64_t site_3, const Complex& coeff) {
        ThreeBodyTransformData td;
        td.op_type_1 = op_type_1;
        td.site_index_1 = site_1;
        td.op_type_2 = op_type_2;
        td.site_index_2 = site_2;
        td.op_type_3 = op_type_3;
        td.site_index_3 = site_3;
        td.coefficient = coeff;
        add_record(td);
    }

    /// Replace this operator's canonical AoS term storage with a verbatim
    /// copy of ``src``'s terms, then invalidate the derived SoA / CSR
    /// caches. Provided so that builders which assemble a fresh operator
    /// from an existing host operator do not have to reach into the
    /// public ``transform_data_`` / ``three_body_data_`` members directly.
    void copyTermsFrom(const Operator& src) {
        transform_data_ = src.transform_data_;
        three_body_data_ = src.three_body_data_;
        extra_terms_ = src.extra_terms_;
        invalidateMatrixCaches();
    }

    /// Invalidate every cache derived from the terms (isReal(), is_hermitian(), the canonical
    /// terms, the row program and the full-space lane with its CSR). Cheap; every mutator calls it.
    virtual void invalidateMatrixCaches() {
        real_check_done_ = false;
        hermitian_check_done_ = false;
        canonical_.reset();
        row_program_.reset();
        lane_.reset();
    }

    uint64_t getNumBits() const { return n_bits_; }
    float getSpin() const { return spin_l_; }

    // -------------------------------------------------------------------
    // LinearOperator interface: dim() / is_hermitian() /
    // description(). apply() is defined with the matvec entry points below.
    // -------------------------------------------------------------------
    [[nodiscard]] std::size_t dim() const override { return static_cast<std::size_t>(1ULL << n_bits_); }
    [[nodiscard]] bool is_hermitian() const override {
        // H^dagger == H on the canonical terms (invariance.h), cached until the terms change.
        if (!hermitian_check_done_) {
            hermitian_cached_ = ed::ops::hermitian(canonical());
            hermitian_check_done_ = true;
        }
        return hermitian_cached_;
    }
    [[nodiscard]] std::string description() const override {
        return "Operator(n_bits=" + std::to_string(n_bits_) + ")";
    }
    /// s_H: the sum of |c| over the canonical terms.
    [[nodiscard]] double norm_bound() const override { return canonical().l1_norm(); }

    Operator(uint64_t n_bits, float spin_l) : n_bits_(n_bits), spin_l_(spin_l) {
        if (n_bits >= 64) {
            throw ed::Unsupported("Operator: n_bits = " + std::to_string(n_bits)
                                  + " >= 64 is not supported (the states are 64-bit words: 1ULL << n_bits)");
        }
    }

    // -------------------------------------------------------------------
    // Copy / move semantics: a copy carries the records; its canonical terms and its
    // full-space lane are rebuilt lazily (so a copy reads ED_CSR_* afresh).
    // -------------------------------------------------------------------
    Operator(const Operator& other)
        : LinearOperator(other), transform_data_(other.transform_data_), three_body_data_(other.three_body_data_),
          extra_terms_(other.extra_terms_), n_bits_(other.n_bits_), spin_l_(other.spin_l_),
          real_check_done_(other.real_check_done_), real_cache_(other.real_cache_) {}

    Operator(Operator&& other) noexcept
        : LinearOperator(std::move(other)), transform_data_(std::move(other.transform_data_)),
          three_body_data_(std::move(other.three_body_data_)), extra_terms_(std::move(other.extra_terms_)),
          n_bits_(other.n_bits_), spin_l_(other.spin_l_), real_check_done_(other.real_check_done_),
          real_cache_(other.real_cache_) {
        other.invalidateMatrixCaches();
    }

    Operator& operator=(const Operator& other) {
        if (this != &other) {
            n_bits_ = other.n_bits_;
            spin_l_ = other.spin_l_;
            transform_data_ = other.transform_data_;
            three_body_data_ = other.three_body_data_;
            extra_terms_ = other.extra_terms_;
            real_check_done_ = other.real_check_done_;
            real_cache_ = other.real_cache_;
            canonical_.reset();
            row_program_.reset();
            lane_.reset();
        }
        return *this;
    }

    Operator& operator=(Operator&& other) noexcept {
        if (this != &other) {
            n_bits_ = other.n_bits_;
            spin_l_ = other.spin_l_;
            transform_data_ = std::move(other.transform_data_);
            three_body_data_ = std::move(other.three_body_data_);
            extra_terms_ = std::move(other.extra_terms_);
            real_check_done_ = other.real_check_done_;
            real_cache_ = other.real_cache_;
            canonical_.reset();
            row_program_.reset();
            lane_.reset();
            other.invalidateMatrixCaches();
        }
        return *this;
    }

    // ========================================================================
    // apply: y = H x on the full 2^N space. Row r is the row walk (row_walk.h) of the
    // program of H^dagger from r, conjugated: <r|H|t> = conj(<t|H^dagger|r>), exact for any
    // H. The rows are assembled once into a CSR when it is allowed, else walked per apply:
    //   ED_CSR_FORCE      1: always the CSR, 0: never (unset: the dimension decides)
    //   ED_CSR_DIM_MAX    the largest dimension assembled (default 2^20)
    // read when the lane is built (the first apply after a change of the records).
    // ========================================================================
    void apply(const Complex* in, Complex* out, std::size_t size) const override {
        const std::uint64_t dim = 1ULL << n_bits_;
        if (size != static_cast<std::size_t>(dim)) {
            throw std::invalid_argument("Operator::apply: input/output vector size mismatch");
        }
        const auto lane = full_space_lane_();
        if (lane->use_csr) {
            lane->csr.spmv(in, out);
            return;
        }
        const auto view = lane->rows->view();
#pragma omp parallel for schedule(static)
        for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
            Complex acc(0.0, 0.0);
            ed::ops::for_each_connection(view, static_cast<std::uint64_t>(ir),
                                         [&](std::uint64_t t, const Complex& h) { acc += std::conj(h) * in[t]; });
            out[ir] = acc;
        }
    }

    /// The representation apply() uses: "csr" or "walk" (builds the lane if needed).
    [[nodiscard]] const char* full_space_lane() const { return full_space_lane_()->use_csr ? "csr" : "walk"; }

    /// H is real in the S^z basis: its canonical terms equal their complex conjugates within
    /// `rtol` of the largest coefficient. Cached until the records change.
    bool isReal(double rtol = ed::ops::kInvarianceRtol) const {
        if (!real_check_done_) {
            real_cache_ = ed::ops::conjugation_invariant(canonical(), rtol);
            real_check_done_ = true;
        }
        return real_cache_;
    }

protected:
    // -------------------------------------------------------------------
    // Lattice constants (protected so derived classes can access).
    // -------------------------------------------------------------------
    uint64_t n_bits_;
    float spin_l_;

    // Caches for ``isReal()`` / ``is_hermitian()``. Invalidated by
    // ``invalidateMatrixCaches()``.
    mutable bool real_check_done_ = false;
    mutable bool hermitian_check_done_ = false;
    mutable bool hermitian_cached_ = true;
    mutable bool real_cache_ = false;

    // The full-space lane: the program apply() walks and, when assembled, its CSR. Built
    // on first use, immutable, reset with the other caches.
    struct FullSpaceLane {
        std::shared_ptr<const ed::ops::MaskedProgram> rows;   // row_program()
        bool use_csr = false;
        ed::matvec::ReducedSymmetryCsr<Complex> csr;
    };
    mutable std::shared_ptr<const FullSpaceLane> lane_;
    mutable std::mutex lane_mutex_;

    [[nodiscard]] std::shared_ptr<const FullSpaceLane> full_space_lane_() const {
        std::lock_guard<std::mutex> lock(lane_mutex_);
        if (lane_) return lane_;
        auto L = std::make_shared<FullSpaceLane>();
        L->rows = row_program();
        const std::uint64_t dim = 1ULL << n_bits_;
        const std::optional<bool> force = ed::env::tristate("ED_CSR_FORCE");
        const long long cut = ed::env::integer("ED_CSR_DIM_MAX", 0);
        const std::uint64_t cutoff = cut > 0 ? static_cast<std::uint64_t>(cut) : (std::uint64_t{1} << 20);
        L->use_csr = dim < (std::uint64_t{1} << 32) && force.value_or(dim <= cutoff);
        if (L->use_csr) assemble_rows_(*L->rows, dim, L->csr);
        lane_ = std::move(L);
        return lane_;
    }

    // The rows of the walk as a CSR (exact zeros dropped), in the walk's order.
    static void assemble_rows_(const ed::ops::MaskedProgram& P, std::uint64_t dim,
                               ed::matvec::ReducedSymmetryCsr<Complex>& csr) {
        const auto view = P.view();
        csr.dim = dim;
        csr.row_ptr.assign(dim + 1, 0);
#pragma omp parallel for schedule(static)
        for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
            std::uint64_t n = 0;
            ed::ops::for_each_connection(view, static_cast<std::uint64_t>(ir), [&](std::uint64_t, const Complex& h) {
                if (h != Complex(0.0, 0.0)) ++n;
            });
            csr.row_ptr[static_cast<std::size_t>(ir) + 1] = n;
        }
        for (std::uint64_t r = 0; r < dim; ++r) csr.row_ptr[r + 1] += csr.row_ptr[r];
        csr.allocate_first_touch();
#pragma omp parallel for schedule(static)
        for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
            std::uint64_t e = csr.row_ptr[static_cast<std::size_t>(ir)];
            ed::ops::for_each_connection(view, static_cast<std::uint64_t>(ir), [&](std::uint64_t t, const Complex& h) {
                if (h == Complex(0.0, 0.0)) return;
                csr.col_idx[e] = static_cast<std::uint32_t>(t);
                csr.val[e] = std::conj(h);
                ++e;
            });
        }
    }

    // The canonical terms (canonical()), immutable once built; reset with the other caches.
    mutable std::shared_ptr<const ed::ops::MaskedOperator> canonical_;
    mutable std::mutex canonical_mutex_;
    mutable std::shared_ptr<const ed::ops::MaskedProgram> row_program_;
    mutable std::mutex row_program_mutex_;

    void check_factor_(uint8_t op, uint64_t site) const {
        if (op > 2)
            throw std::invalid_argument("Operator: op type " + std::to_string(op) + " is not 0 (S+), 1 (S-) or 2 (Sz)");
        if (site >= n_bits_)
            throw std::invalid_argument("Operator: site " + std::to_string(site) + " is outside [0, "
                                        + std::to_string(n_bits_) + ")");
    }
};
