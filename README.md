# plenoptomos

Plenoptic imaging reconstruction tools, developed in the context of the VOXEL
project, part of the European Union's Horizon 2020 research and innovation
programme ([VOXEL H2020-FETOPEN-2014-2015-RIA  GA 665207](https://ec.europa.eu/programmes/horizon2020/en/news/3d-x-ray-imaging-very-low-dose)).

It provides a Python package that allows to read and manipulate light-field images.
The provided features are:
* support for a few types of datasets and formats (including the .vox format).
* support for refocusing using standard algorithms like the integration [3] and Fourier [4] methods.
* support for GPU accelerated tomography based refocusing, including the following algorithms:
  * unfiltered back-projection [1]
  * SIRT [2]
  * Chambolle-Pock (Least-square, TV) [2]
* support for depth estimation [5]

This work is based on the following articles:
- [1] N. Viganò, et al., “Tomographic approach for the quantitative scene reconstruction from light field images,” Opt. Express, vol. 26, no. 18, p. 22574, Sep. 2018.
- [2] N. Viganò, et al., “Advanced light-field refocusing through tomographic modeling of the photographed scene,” Opt. Express, vol. 27, no. 6, p. 7834, Mar. 2019.

It also implements methods from, the following articles:
- [3] R. Ng, et al., “Light Field Photography with a Hand-held Plenoptic Camera,” Stanford Univ. Tech. Rep. CSTR 2005-02, 2005.
- [4] R. Ng, “Fourier slice photography,” ACM Trans. Graph., vol. 24, no. 3, p. 735, 2005.
- [5] M. W. Tao, et al., “Depth from combining defocus and correspondence using light-field cameras,” Proceedings of the IEEE International Conference on Computer Vision, 2013, pp. 673–680.

Other useful information:
* Free software: GNU General Public License v3
* Documentation: [https://cicwi.github.io/plenoptomos]

## Getting Started

It takes a few steps to setup plenoptomos on your
machine. We recommend installing
[Anaconda package manager](https://www.anaconda.com/download/) for
Python 3.

### Installing with conda

Simply install with:
```
conda install -c cicwi plenoptomos
```

### Installing from source

To install plenoptomos, simply clone this GitHub
project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/cicwi/plenoptomos.git
cd plenoptomos
pip install -e .
```

### Running the examples

To learn more about the functionality of the package check out our
examples folder.

## Authors and contributors

* **Nicola Viganò** - *Main developer*
* **Francesco Brun** - *Initial contributions*
* **Pablo Martinez Gil** - *Contributed to the creation and enabling of the .vox data format, and to the wavelet solver*
* **Charlotte Herzog** - *Contributed to the creation and enabling of the .vox data format*

See also the list of [contributors](https://github.com/cicwi/plenoptomos/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `master` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.

## Data format

This package comes with a suggested data format for the light-fields.
We refer to the [File format](https://cicwi.github.io/plenoptomos) section of the documentation for more information.