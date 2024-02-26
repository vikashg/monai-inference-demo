from handler import ModelHandler 
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext
from pathlib import Path
import os 

MODEL_PT_FILE = '/home/gupta/disk/Tools/Vikash/monai-models/breast_density_classification/torch-serve/traced_ts_model.pt'
CURR_FILE_PATH=Path(__file__).parent.absolute()
EXAMPLE_ROOT_DIR=CURR_FILE_PATH
TEST_DATA='/home/gupta/disk/Tools/Vikash/monai-models/breast_density_classification/torch-serve/data/1.2.840.113713.17.1538.156204584798095915006091931534607431491.jpg'
def test_handler(batch_size=1):
	handler = ModelHandler()
	print(EXAMPLE_ROOT_DIR.as_posix())
	ctx = MockContext(model_pt_file = MODEL_PT_FILE, 
						model_dir = EXAMPLE_ROOT_DIR.as_posix(),
						model_file = None,)
	handler.initialize(ctx)
	handler.context = ctx
	handler.handle(TEST_DATA, ctx)

if __name__ == '__main__':
	test_handler()
