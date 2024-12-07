import os
import pytest
from click.testing import CliRunner
from sygnals_nn.cli import cli

@pytest.fixture
def runner():
    """Fixture to provide a Click test runner."""
    return CliRunner()

def test_cli_help(runner):
    """Test that the main CLI command `sygnals-nn --help` prints usage information."""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "Usage: " in result.output
    assert "Commands:" in result.output
    assert "create" in result.output
    assert "train" in result.output
    assert "run" in result.output
    assert "export" in result.output

def test_cli_create_missing_args(runner):
    """
    Test that the `create` command fails when required arguments are not provided.
    For example, if `--layers` is missing.
    """
    result = runner.invoke(cli, ['create', '--activation', 'relu'])
    assert result.exit_code != 0
    assert "Error: Missing option '--layers'" in result.output

def test_cli_create_success(runner, tmp_path):
    """
    Test a successful `create` command run.
    This test checks that the command runs without error and that the output model file is created.
    """
    model_file = tmp_path / "test_model.keras"
    result = runner.invoke(cli, [
        'create',
        '--layers', '4,8,1',
        '--activation', 'relu,sigmoid',
        '--loss', 'binary_crossentropy',
        '--optimizer', 'adam',
        '--output', str(model_file)
    ])
    assert result.exit_code == 0
    assert "Model saved to" in result.output
    assert model_file.exists()

def test_cli_train_missing_args(runner):
    """Test that the `train` command fails if required arguments are missing."""
    result = runner.invoke(cli, ['train'])
    # Expect failure due to missing required arguments
    assert result.exit_code != 0
    assert "Error: Missing option '--model'" in result.output

def test_cli_run_missing_args(runner):
    """Test that the `run` command fails if required arguments are missing."""
    result = runner.invoke(cli, ['run'])
    assert result.exit_code != 0
    assert "Error: Missing option '--model'" in result.output

def test_cli_export_missing_args(runner):
    """Test that the `export` command fails if required arguments are missing."""
    result = runner.invoke(cli, ['export'])
    assert result.exit_code != 0
    assert "Error: Missing option '--predictions'" in result.output

