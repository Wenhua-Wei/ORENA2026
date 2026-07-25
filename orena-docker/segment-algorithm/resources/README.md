# `resources/` — your model lives here

Everything in this folder is **copied into the container image** at build time and
is available at inference. This is where your trained model belongs.

| File | What it is | Replace it? |
|:--|:--|:--|
| `model.py` | the model definition (`DummyModel`, a trivial CNN) | **yes** — with your architecture |
| `dummy_weights.pt` | a tiny checkpoint so the template runs out of the box | **yes** — with your trained weights |

`inference.py` imports from here:

```python
from resources.model import DummyModel

WEIGHTS_PATH = RESOURCES_PATH / "dummy_weights.pt"
```

Keep that wiring, or point it at whatever you name your own files.

## Things to get right

**There is no internet at inference time.** The container runs with networking
disabled, so nothing can be downloaded at runtime — no model hubs, no checkpoints,
no tokenizer files, no fonts. Everything your model touches must be inside the
image. If you use a library that lazily fetches weights on first call (many
`transformers`, `timm`, and `open_clip` entry points do), pre-download during the
Docker build and load from a local path instead.

**Load your model once.** One container run answers a whole batch of questions, so
the checkpoint is loaded once and reused across every question in that batch. Keep
model loading outside the per-question loop in `inference.py` — that is where the
setup allowance in the latency budget comes from.

**Large checkpoints need splitting.** A single image layer cannot exceed **50 GB**,
and each Docker `COPY` is one layer. If your weights are bigger than that, split
them into chunks, copy them with several `COPY` instructions, and reassemble at
runtime — see the comment in the `Dockerfile`.

**Big weights make big images.** The whole image is pulled before your code starts.
That pull is infrastructure overhead and is not charged against your latency
budget, but it does slow every submission cycle, so keep the image only as large as
it needs to be.
