[![pypi](https://img.shields.io/pypi/v/skrl)](https://pypi.org/project/skrl)
[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20models-hugging%20face-F8D521">](https://huggingface.co/skrl)
![discussions](https://img.shields.io/github/discussions/Toni-SM/skrl)
<br>
[![license](https://img.shields.io/github/license/Toni-SM/skrl)](https://github.com/Toni-SM/skrl)
<span>&nbsp;&nbsp;&nbsp;&nbsp;</span>
[![docs](https://readthedocs.org/projects/skrl/badge/?version=latest)](https://skrl.readthedocs.io/en/latest/?badge=latest)
[![pytest](https://github.com/Toni-SM/skrl/actions/workflows/python-test.yml/badge.svg)](https://github.com/Toni-SM/skrl/actions/workflows/python-test.yml)
[![pre-commit](https://github.com/Toni-SM/skrl/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/Toni-SM/skrl/actions/workflows/pre-commit.yml)

<br>
<p align="center">
  <img width="300rem" src="https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/_static/data/skrl-up-transparent.png">
</p>
<h2 align="center" style="border-bottom: 0 !important;">SKRL - Reinforcement Learning library</h2>
<br>

**skrl** is an open-source modular library for Reinforcement Learning written in Python (using [PyTorch](https://pytorch.org/)) and designed with a focus on readability, simplicity, and transparency of algorithm implementation. In addition to supporting the OpenAI [Gym](https://www.gymlibrary.dev) / Farama [Gymnasium](https://gymnasium.farama.org) and [DeepMind](https://github.com/deepmind/dm_env) and other environment interfaces, it allows loading and configuring [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym/), [NVIDIA Isaac Orbit](https://isaac-orbit.github.io/orbit/index.html) and [NVIDIA Omniverse Isaac Gym](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_isaac_gym.html) environments, enabling agents' simultaneous training by scopes (subsets of environments among all available environments), which may or may not share resources, in the same run

<br>

### Please, visit the documentation for usage details and examples

https://skrl.readthedocs.io/en/latest/

<br>

> **Note:** This project is under **active continuous development**. Please make sure you always have the latest version. Visit the [develop](https://github.com/Toni-SM/skrl/tree/develop) branch or its [documentation](https://skrl.readthedocs.io/en/develop) to access the latest updates to be released.

<br>

### Citing this library

To cite this library in publications, please use the following reference:

```bibtex
@article{serrano2022skrl,
  title={skrl: Modular and Flexible Library for Reinforcement Learning},
  author={Serrano-Mu{\~n}oz, Antonio and Arana-Arexolaleiba, Nestor and Chrysostomou, Dimitrios and B{\o}gh, Simon},
  journal={arXiv preprint arXiv:2202.03825},
  year={2022}
}
```
