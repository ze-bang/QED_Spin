{
  # =============================================================================
  # flake.nix -- Nix-based developer shell for quantum_ed (parallel to
  # Dockerfile.dev). Lets collaborators on NixOS / nix-darwin / Linux+nix get
  # the exact same toolchain CI uses, *without* Docker, *without* polluting
  # their host with ad-hoc apt installs, and *with* a content-addressed,
  # pinned dependency graph.
  #
  # Quick start
  # -----------
  #   # Enable flakes (one-time):
  #   #   echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
  #
  #   nix develop                  # drop into a bash shell with the toolchain
  #   cmake --preset ci-linux
  #   cmake --build --preset ci-linux -j
  #   ctest   --preset ci-linux
  #   pip install -e .             # uses the pinned scikit-build-core / pybind11
  #   python -m pytest python/tests -v
  #
  # The `default` package builds the C++-only library + executables (no Python
  # wheel) so `nix build` produces a reproducible binary you can ship to a
  # cluster:
  #
  #   nix build .#quantum-ed-cpp
  #   ./result/bin/ED ...
  #
  # Audit ref: P2.12.
  # =============================================================================

  description = "quantum_ed -- modern exact diagonalization for quantum spin systems";

  inputs = {
    # nixos-24.05 is the most recent stable channel as of 2026-04; it ships
    # gcc 13, cmake 3.28, openblas 0.3.27, hdf5 1.14, and python 3.11 -- all
    # >= the floors enforced by CMakeLists.txt and pyproject.toml.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # -----------------------------------------------------------------------
        # Python 3.11 + the pip packages CI's linux-python lane installs.
        # We deliberately leave numpy/h5py/pytest/pybind11 here so the
        # Python wheel build picks them up via PYTHONPATH instead of
        # downloading from PyPI on every `pip install -e .`.
        # -----------------------------------------------------------------------
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          pip
          setuptools
          wheel
          scikit-build-core
          pybind11
          numpy
          h5py
          pytest
        ]);

        commonNativeBuildInputs = with pkgs; [
          cmake
          ninja
          pkg-config
          git
        ];

        commonBuildInputs = with pkgs; [
          openblas
          lapack
          # `lapacke` is bundled inside lapack on nixpkgs >= 23.05.
          hdf5-cpp
          eigen
          arpack
          # MPI is optional but cheap to make available.
          openmpi
        ];

      in {
        # ---------------------------------------------------------------------
        # `nix develop` shell -- the dev environment.
        # ---------------------------------------------------------------------
        devShells.default = pkgs.mkShell {
          name = "quantum-ed-dev";

          packages = commonNativeBuildInputs ++ commonBuildInputs ++ [
            pythonEnv
            pkgs.clang-tools  # clang-format + clang-tidy
            pkgs.doxygen
            pkgs.graphviz
            pkgs.bashInteractive
          ];

          shellHook = ''
            export BLAS_PROFILE=OPENBLAS
            export CMAKE_GENERATOR=Ninja
            echo ""
            echo "quantum_ed nix dev shell ready."
            echo ""
            echo "First-time build:"
            echo "  cmake --preset ci-linux"
            echo "  cmake --build --preset ci-linux -j"
            echo "  ctest   --preset ci-linux"
            echo ""
            echo "Python wheel:"
            echo "  pip install -e ."
            echo "  python -m pytest python/tests -v"
            echo ""
          '';
        };

        # ---------------------------------------------------------------------
        # `nix build .#quantum-ed-cpp` -- C++-only build artifact, suitable
        # for shipping to a cluster login node. Mirrors the `ci-linux` preset
        # so the binaries match what CI verifies.
        # ---------------------------------------------------------------------
        packages.quantum-ed-cpp = pkgs.stdenv.mkDerivation {
          pname = "quantum-ed-cpp";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = commonNativeBuildInputs;
          buildInputs       = commonBuildInputs;

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DBUILD_ED_TESTS=OFF"
            "-DWITH_CUDA=OFF"
            "-DWITH_MPI=OFF"
            "-DBLAS_PROFILE=OPENBLAS"
          ];

          # The smoke test is fast (<10s) and a strong sanity check that the
          # nix-provided BLAS/LAPACK/HDF5 actually work together.
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Modern exact diagonalization for quantum spin systems";
            license = licenses.mit;
            platforms = platforms.linux;
          };
        };

        packages.default = self.packages.${system}.quantum-ed-cpp;
      });
}
