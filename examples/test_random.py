from typing import Optional
import numpy as np

#np.random.seed(0)

def random_seq(rng):
    return rng.integers(21)

class RandomNumber(object):
    def __init__(
        self,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.rng = np.random.default_rng(seed)
    
    def generate(self):
        rand_nums1 = self.rng.random(3)
        rand_nums2 = random_seq(self.rng)
        return rand_nums1, rand_nums2

seed = 42

generator = RandomNumber(seed)

for i in range(3):
    rand_nums = generator.generate()
    print(rand_nums)
