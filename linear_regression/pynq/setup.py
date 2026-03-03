import numpy as np
from pynq import Overlay
import time
from pynq import allocate

ol = Overlay("outer_product_2.1_o.bit")
ip = ol.outer_product_accum_0