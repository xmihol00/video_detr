
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import unittest
from cnnSearch.models.supernet import ResNetSuperNet
from cnnSearch.models.subnet import extractSubnetFromSupernet
from cnnSearch.search_space import sampleRandomArchitecture, DEFAULT_SEARCH_SPACE

class TestSubnetExtraction(unittest.TestCase):
    def test_output_consistency(self):
        # 1. Initialize SuperNet
        supernet = ResNetSuperNet(DEFAULT_SEARCH_SPACE)
        supernet.eval()

        # 2. Sample random architecture
        config = sampleRandomArchitecture(DEFAULT_SEARCH_SPACE)
        # Ensure aux heads are disabled for comparison simplicity if Subnet doesn't support them
        # But Subnet is just the backbone + head. Supernet returns (logits, config, aux_logits).
        # We compare main logits.
        
        # 3. Create input
        input_tensor = torch.randn(1, 3, config.inputResolution, config.inputResolution)

        # 4. Run SuperNet
        with torch.no_grad():
            super_logits, _, _ = supernet(input_tensor, config)

        # 5. Extract Subnet
        extracted = extractSubnetFromSupernet(supernet, config)
        subnet = extracted.model
        subnet.eval()

        # 6. Run Subnet
        with torch.no_grad():
            sub_logits = subnet(input_tensor)

        # 7. Compare
        # We expect very close values. SlimConv2d does slicing, Subnet does standard conv.
        # Weights are copied.
        
        # Check max difference
        diff = (super_logits - sub_logits).abs().max()
        print(f"Max difference: {diff.item()}")
        
        self.assertTrue(torch.allclose(super_logits, sub_logits, atol=1e-5), 
                        f"Logits mismatch. Max diff: {diff.item()}")

if __name__ == '__main__':
    unittest.main()
