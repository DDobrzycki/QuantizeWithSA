
# QuantizeWithSA

Welcome to our PTQ repository, featuring a flexible quantization tool utilizing the Simulated Annealing algorithm. This tool empowers users to quantize their desired models with fixed-point precision, optimizing performance while preserving accuracy. Customize simulations, convergence steps, and search ranges for fractional bits, ensuring tailored quantization results. Control degradation thresholds and convergence guidance through adjustable hyperparameters for fine-tuning. Explore our repository and unlock efficient model compression with ease.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Using pip

1. Clone the repository:
    ```sh
    git clone https://github.com/DDobrzycki/QuantizeWithSA.git
    cd QuantizeWithSA
    ```

2. Create a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Using conda

1. Clone the repository:
    ```sh
    git clone https://github.com/DDobrzycki/QuantizeWithSA.git
    cd QuantizeWithSA
    ```

2. Create a conda environment from the provided environment file:
    ```sh
    conda env create -f environment.yml
    ```

3. Activate the conda environment:
    ```sh
    conda activate QuantizeWithSA
    ```

After completing these steps, the required dependencies will be installed, and you can start using the PTQ quantizer.

## Usage

To run the project, use the following command:
```bash

```

## Examples

...

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes.
4. Push your branch: `git push origin feature-name`.
5. Create a pull request.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- Non-Commercial — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

See the [LICENSE](./LICENSE.txt) file for more details.
