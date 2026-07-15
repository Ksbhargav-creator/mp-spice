# ---- builder ----
FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY mp-spice/ mp-spice/
COPY mtl5/ mtl5/

RUN echo "--- /src/mp-spice contents ---" && ls -la mp-spice/ && \
    echo "--- /src/mtl5 contents ---" && ls -la mtl5/

# The file currently references a local checkout of mtl5
# This will be solved when the local checkout gets merged
# or when the kernel_stats is moved to mp-spice
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
    libstdc++6 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/mp-spice/build/applications/klu_quire_IR_study/klu_quire_IR_study /usr/local/bin/
COPY --from=builder /src/mp-spice/build/applications/klu_quire_study/klu_quire_study /usr/local/bin/

WORKDIR /work
LABEL authors="shreebhargavkomanabelli"
ENTRYPOINT ["klu_quire_IR_study"]
