# Third-party notices

This repository contains an independent JAX adaptation informed by the
following projects. Their source trees are not vendored into this repository.
The setup script checks out the exact PureJaxRL and Banyan revisions into the
external runtime source directory; the loss-of-plasticity revision is recorded
as the provenance for the JAX reimplementation and is not needed at runtime.

## PureJaxRL

- Source: https://github.com/luchris429/purejaxrl
- Revision: `31756b197773a52db763fdbe6d635e4b46522a73`
- License: Apache License 2.0
- Use here: recurrent PPO rollout, GAE, recurrent trajectory minibatching,
  clipped PPO loss, and update structure, ported for Banyan. The network
  topology and initialization follow the implementation separately supplied
  by the experiment's authors.

Copyright and license terms remain those stated by the PureJaxRL project. The
Apache License 2.0 text is available at
https://github.com/luchris429/purejaxrl/blob/31756b197773a52db763fdbe6d635e4b46522a73/LICENSE.

## Banyan

- Source: https://github.com/nhshah15/banyan
- Revision: `7bb5d385cb2b48f49ab01d0c9a47b9d404e2ecc3`
- License: Apache License 2.0
- Use here: environment, ruleset codec, task generator, task/layout helpers,
  and fixed combination mapping. Installed as a non-editable snapshot of the
  exact clean revision by the cluster setup script.

Copyright and license terms remain those stated by the Banyan project. The
Apache License 2.0 text is available at
https://github.com/nhshah15/banyan/blob/7bb5d385cb2b48f49ab01d0c9a47b9d404e2ecc3/LICENSE.

## Loss of Plasticity

- Source: https://github.com/shibhansh/loss-of-plasticity
- Revision: `a6b79580d85f3025bdb601566d3627c5f489f13b`
- License: MIT
- Copyright: 2024 Shibhansh Dohare
- Use here: generate-and-test CBP mechanics, reimplemented in JAX.

The MIT license text and copyright notice are available at
https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/LICENSE.
