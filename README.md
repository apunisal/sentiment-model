# ✨ Vibe Check: AI Sentiment Analysis Web App ✨

An interactive, user-friendly AI application built completely from scratch using PyTorch, exported via ONNX, containerized, and deployed to an enterprise OpenShift cluster.

### 🎨 The Interface (Ready for Input)
![Vibe Check Main Screen](Screenshot%202026-07-02%20at%2012.15.04%E2%80%AFPM.png)

---

## 1. What the Model Is & What It's For
This is a custom **Sentiment Analysis Neural Network** designed to perform a real-time "vibe check" on text inputs.

* **The Goal:** It processes user-submitted English phrases and classifies them into one of two emotional categories: **Positive Vibe** or **Negative Vibe**.
* **Advanced Context Parsing:** Unlike basic keyword-matching algorithms, this model utilizes structural bigram tokenization. This prevents it from being fooled by simple negation configurations. It accurately understands that while `"bad"` is inherently negative, contextual phrases like `"not bad"`, `"never terrible"`, or `"no problem"` represent positive outcomes.
* **Large Lexicon Capacity:** The neural network is trained using an algorithmic cross-product loop mapping out **37,604 distinct sentence combinations** spanning standard, premium, executive, and chaotic semantic classifications.

---

## 2. Live Inference Examples

Here is the model successfully analyzing different types of phrasing in real-time:

**Standard Positive Recognition:**
![Positive Inference Demo](Screenshot%202026-07-02%20at%2012.15.24%E2%80%AFPM.png)

**Advanced Negation Handling ("not bad" = Positive):**
![Advanced Negation Demo](Screenshot%202026-07-02%20at%2012.47.06%E2%80%AFPM.png)

---

## 3. How to Use This in OpenShift

Once successfully built and deployed, the application bypasses raw developer JSON endpoints to offer a highly responsive web interface accessible via a public OpenShift Route.

### Prerequisites
Ensure your configuration file (`Deployment.yaml`) matches the clean `v1` tag definition with a forced download policy to ignore local container layer caching:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-model-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sentiment-model
  template:
    metadata:
      labels:
        app: sentiment-model
    spec:
      containers:
      - name: sentiment-model
        image: docker.io/apunisal/sentiment-model:v1
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
```

### Deployment Commands
To spin up your live endpoint on your cluster, execute the following commands in your cluster terminal session:

```bash
# 1. Apply the deployment configuration to the cluster
oc apply -f Deployment.yaml

# 2. Force the pods to pull the freshest v1 image and restart immediately
oc rollout restart deployment/sentiment-model-deployment

# 3. Track rollout progress until successful initialization
oc get pods -w
```

Once running, simply navigate to your public OpenShift Route URL in any desktop or mobile browser to run live sentiment checks.

---

## 4. How It Was Created: The Full Engineering Process

### Step 1: Neural Network Architecture & Tokenization Pipeline
We built a custom PyTorch classification network (`SentimentModel`) consisting of an embedding layer (`nn.Embedding`), a dense layer featuring ReLU activation, and a final linear layer outputting through a `Sigmoid` activation function to scale certainty scores cleanly between 0 and 1.

To ensure consistent text parsing between training scripts and production web routes, we designed a custom tokenizer using `hashlib.md5`. When it encounters negation markers (`not`, `no`, `never`, `dont`, etc.), it binds them to the subsequent token (e.g., `"not" + "bad" = "not_bad"`), mapping it cleanly into a stable `12,000` integer vocabulary allocation table.

### Step 2: Training on Algorithmic Dictionary Matrices
The model was fed an advanced algorithmic training set of **37,604 sample combinations**, blending random context starters with standard, premium, executive, and chaotic semantic classifications. The system optimized quickly over 40 epochs using a Binary Cross-Entropy loss criterion (`nn.BCELoss`) combined with an Adam Optimizer, bringing the final training loss down to an optimized state of `0.2504`.

### Step 3: Lightweight ONNX Core Model Export
To make the model deployment-ready for lightning-fast inference outside of a bloated PyTorch environment, the trained weights were exported into a universal format using the **ONNX (Open Neural Network Exchange)** specification at Opset version 18:

```python
torch.onnx.export(model, dummy_input, "sentiment_model.onnx", opset_version=18)
```

This outputted `sentiment_model.onnx` along with its auxiliary structural weight asset `sentiment_model.onnx.data`.

### Step 4: Production API Serving and Web UI Integration
Using **FastAPI** and **Uvicorn**, we built `app.py` to serve a dual purpose. It handles rapid POST incoming payloads by mapping text against the `onnxruntime` engine, while simultaneously outputting a responsive, custom styled frontend interface decorated with floating micro-animations at its base path response (`/`).

### Step 5: Container Execution
The workspace was compiled utilizing a specialized multi-layer clean build execution configuration:

```bash
podman build --no-cache -t docker.io/apunisal/sentiment-model:v1 .
podman push docker.io/apunisal/sentiment-model:v1
```
