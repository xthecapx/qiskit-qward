# Contributing

**We appreciate all kinds of help, so thank you!**

## Contributing to Quantum Prototype Template

Specific details for contributing to this project are outlined below.

### Setting Up the Development Environment

To get started with local development:

1. Clone the repository and navigate to the project root directory
2. Install the package in development mode:
   ```shell
   pip install -e .
   ```
   This installs the package in "editable" mode, allowing you to make changes without reinstalling.

3. Set up your IBM Quantum credentials:
   - Copy the `.env.example` file to `.env`
     ```shell
     cp .env.example .env
     ```
   - Edit the `.env` file with your credentials:
     ```
     # IBM Quantum credentials
     IBM_QUANTUM_CHANNEL=ibm_quantum
     IBM_QUANTUM_TOKEN=your_token_here
     ```
   - You can obtain your IBM Quantum token from the [IBM Quantum Experience Dashboard](https://quantum-computing.ibm.com/).
   - Note: Keep your `.env` file private and never commit it to the repository.

4. Alternatively, use Docker to set up a complete development environment:
   ```shell
   chmod +x start.sh
   ./start.sh
   ```
   This script builds and starts a Docker container with all dependencies installed, and opens a Jupyter Lab interface at http://127.0.0.1:4321/lab.

### Reporting Bugs and Requesting Features

Users are encouraged to use GitHub Issues for reporting issues and requesting features.

### Ask/Answer Questions and Discuss Quantum Prototype Template

Users are encouraged to use GitHub Discussions for engaging with researchers, developers, and other users regarding this quantum prototype and the provided examples.

### Project Code Style

Code in this repository should conform to PEP8 standards. Style/lint checks are run to validate this. Line length must be limited to no more than 88 characters.

### Pull Request Checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the _code style_ of this project and successfully
   passes the _unit tests_. This prototype uses [Pylint](https://www.pylint.org) and
   [PEP8](https://www.python.org/dev/peps/pep-0008) style guidelines.

   You can run
   ```shell script
   tox -elint
   ```
   from the root of the repository clone for lint conformance checks.
