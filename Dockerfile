FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt && \
    rm requirements.txt

RUN pip uninstall numpy -y
RUN pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple "numpy<2.0"

RUN python -c "import cv2, torch, numpy"