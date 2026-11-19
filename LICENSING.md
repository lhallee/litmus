# Licensing

Litmus is free for any noncommercial use, and its code, model, data recipe and evaluation are all published so
that anyone can see how a reading and its uncertainty are made. Commercial use is not licensed.

| What | Terms |
| --- | --- |
| The code (engine, app, training, corpus and benchmark pipelines) | [PolyForm Noncommercial 1.0.0](LICENSE) |
| The model: weights, reference bank, calibration tables | PolyForm Noncommercial 1.0.0 |
| The documentation and figures | CC BY-NC 4.0 |
| The corpus | Not released: the compiled corpus is a private dataset. Its recipe is public (the source catalog in `corpus/sources.csv`, the pipeline and the split), so anyone can rebuild it from the sources, each under its own terms. If it is released, every row keeps its source's terms and the compilation itself is CC BY-NC 4.0. |
| Text submitted to the contribute site | Never published or released: it is kept privately to improve the models |
| Bundled third-party parts (the ModernBERT tokenizer, Rust crates, fonts) | Their own licenses, listed in `engine/THIRD_PARTY.md` |

## What that means in practice

- **Allowed:** using, studying, changing and sharing Litmus for any noncommercial purpose. That includes personal
  use, research, and use by schools, universities, charities, public research bodies and government institutions,
  whatever their funding.
- **Required when you share it:** pass along the license (its text or its address) and the `Required Notice:` line
  at the top of `LICENSE`.
- **Not allowed:** commercial use, such as a company running it on its own or its customers' text, or selling a
  product or service built on it.

Litmus is open in the plain sense: everything is published. It is not "open source" as the Open Source Initiative
defines the term, because that definition does not permit a restriction on commercial use. Describe it as source
available and free for noncommercial use.
