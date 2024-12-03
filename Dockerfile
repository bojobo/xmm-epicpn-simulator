FROM bojobo/sas:21.0.0 AS builder

USER 0

RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install beautifultable h5py loguru lxml pydantic pypdf python-dotenv yt pre-commit

FROM builder

# Add SIXTE
COPY --from=bojobo/sixte --chown=heasoft:heasoft /opt/simput /opt/simput
ENV SIMPUT=/opt/simput \
    SIXTE=/opt/simput \
    PATH=/opt/simput/bin:${PATH} \
    PFILES=${PFILES}:/opt/simput/share/sixte/pfiles:/opt/simput/share/simput/pfiles \
    LD_LIBRARY_PATH=/opt/simput/lib:${LD_LIBRARY_PATH}

USER heasoft
