
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest

# Add the 'scripts' directory to the path to allow importing train_model
scripts_dir = Path(__file__).parent.parent / 'scripts'
sys.path.append(str(scripts_dir))

@patch('train_model.prepare_data')
@patch('train_model.AerosolModel')
@patch('train_model.DataLoader')
@patch('train_model.compute_fisher_information')
def test_main_function(mock_fisher, mock_dataloader, mock_model, mock_prepare_data):
    """
    Tests the main function of the training script to ensure it correctly
    parses arguments and executes the training workflow.
    """
    import train_model
    # 1. Setup mocks
    mock_prepare_data.return_value = (MagicMock(), MagicMock())
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance

    # Mock DataLoader to return a single dummy batch with mock tensors
    mock_tensor = MagicMock()
    mock_dataloader.return_value = [{'input': mock_tensor, 'target': mock_tensor}]

    # Mock the criterion to return a mock loss object
    mock_loss = MagicMock()
    mock_loss.item.return_value = 0.0
    mock_model_instance.criterion.return_value = mock_loss

    # 2. Define arguments for the script
    test_args = [
        'train_model.py',
        '--start-date', '20250101',
        '--end-date', '20250110',
        '--output-dir', '/tmp/test_output',
        '--epochs', '1',
        '--batch-size', '16'
    ]

    # 3. Run the main function with patched sys.argv
    with patch.object(sys, 'argv', test_args):
        train_model.main()

    # 4. Assert that the key functions were called correctly
    mock_prepare_data.assert_called_once()
    assert mock_prepare_data.call_args[0][0] == '20250101'
    assert mock_prepare_data.call_args[0][1] == '20250110'
    mock_model.assert_called_once()
    assert mock_dataloader.call_count == 2
    mock_model_instance.save.assert_called()

if __name__ == "__main__":
    pytest.main(['-v', __file__])
