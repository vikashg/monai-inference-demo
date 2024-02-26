from handler import ModelHandler
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext
from pathlib import Path
import os

MODEL_PT_FILE='traced_segres_model.pt'
CURR_FILE_PATH = Path(__file__).parent.absolute()
EXAMPLE_ROOT_DIR=CURR_FILE_PATH
TEST_DATA=os.path.join(CURR_FILE_PATH, 'test.nii.gz')


def test_segresnet(batch_size=1):
	handler = ModelHandler()
	print(EXAMPLE_ROOT_DIR.as_posix())
	ctx = MockContext(model_pt_file = MODEL_PT_FILE, 
								model_dir= EXAMPLE_ROOT_DIR.as_posix(), 
								model_file = None,)
	handler.initialize(ctx)
	handler.context = ctx 
	handler.handle(TEST_DATA, ctx)

if __name__ == '__main__':
	test_segresnet()
