# ---- builder ----
FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY mp-spice/ mp-spice/
COPY mtl5/ mtl5/

RUN echo "--- /src/mp-spice contents ---" && ls -la mp-spice/ && \
    echo "--- /src/mtl5 contents ---" && ls -la mtl5/

# NOTE: this still requires a local mtl5 checkout (FETCHCONTENT_SOURCE_DIR_MTL5),
# NOT the vanilla GitHub FetchContent path from mp-spice's own README. As of this
# writing, `include/mtl/sparse/instrumentation/kernel_stats.hpp` and the
# accumulator-seam changes to native_klu.hpp/sparse_lu.hpp/triangular_solve.hpp
# that mp-spice's quire studies depend on are local, uncommitted/unpushed changes
# on the `quire-instrumentation` branch of the mtl5 checkout -- not yet on
# origin/main. Once that branch is merged upstream, this can drop the local
# checkout and match the README's plain `cmake -B build` path.
#
# Builds every application under applications/ (klu_quire_IR_study,
# klu_quire_study, klu_precision_study, klu_mixed_precision_study,
# klu_mixed_precision) -- omitting a --target builds the default `all` target.
RUN cmake -B mp-spice/build -S mp-spice \
      -DCMAKE_BUILD_TYPE=Release \
      -DMPSPICE_MIXED_PRECISION_KLU=ON \
      -DMPSPICE_BUILD_TESTS=OFF \
      -DFETCHCONTENT_SOURCE_DIR_MTL5=/src/mtl5 \
      -Wno-dev && \
    cmake --build mp-spice/build -j"$(nproc)"

# ---- runtime ----
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 python3 python3-pandas python3-matplotlib bash curl ca-certificates tar gzip && \
    rm -rf /var/lib/apt/lists/*

# All five study binaries -- see applications/*/CMakeLists.txt for what each does.
COPY --from=builder /src/mp-spice/build/applications/klu_quire_IR_study/klu_quire_IR_study /usr/local/bin/
COPY --from=builder /src/mp-spice/build/applications/klu_quire_study/klu_quire_study /usr/local/bin/
COPY --from=builder /src/mp-spice/build/applications/klu_precision_study/klu_precision_study /usr/local/bin/
COPY --from=builder /src/mp-spice/build/applications/klu_mixed_precision_study/klu_mixed_precision_study /usr/local/bin/
COPY --from=builder /src/mp-spice/build/applications/klu_mixed_precision/klu_mixed_precision /usr/local/bin/

# Sweep/data tooling -- lets the container reproduce the matrix sweep and
# csv normalization end to end, not just run one binary against one matrix.
COPY --from=builder /src/mp-spice/scripts/ /opt/mp-spice/scripts/
COPY --from=builder /src/mp-spice/csv/normalize_dynamic_range.py /opt/mp-spice/csv/
COPY --from=builder /src/mp-spice/csv/plot_dynamic_range.py /opt/mp-spice/csv/
RUN chmod +x /opt/mp-spice/scripts/*.sh && \
    ln -s /opt/mp-spice/scripts/run_matrix_sweep.sh /usr/local/bin/run_matrix_sweep.sh && \
    ln -s /opt/mp-spice/scripts/fetch_matrices.sh /usr/local/bin/fetch_matrices.sh

# Regression-check bundle -- the entrypoint the K8s Job/CronJob manifests
# (see k8s/) invoke. baseline.json is committed to the repo and only ever
# updated by a human via PR (see ci/regression_check.py's design note on
# why it must never auto-update from a passing run).
COPY --from=builder /src/mp-spice/ci/regression_check.py /opt/mp-spice/ci/
COPY --from=builder /src/mp-spice/ci/baseline.json /opt/mp-spice/ci/
COPY --from=builder /src/mp-spice/run_regression.sh /opt/mp-spice/run_regression.sh
RUN chmod +x /opt/mp-spice/run_regression.sh

# /work is meant to be a mounted volume (e.g. `-v $PWD:/work`) so downloaded
# matrices, logs/, and csv/ output land on the host, not inside the container.
WORKDIR /work
LABEL authors="shreebhargavkomanabelli"

# No fixed ENTRYPOINT since there are now five binaries plus the sweep script --
# invoke whichever you need, e.g.:
#   docker run -v $PWD:/work IMAGE klu_quire_IR_study data/rajat13/rajat13.mtx
#   docker run -v $PWD:/work IMAGE run_matrix_sweep.sh rajat13 rajat19
CMD ["/bin/bash"]
