"""
Requires:

exiftool
convert (ImageMagick)

Example usage:

python unpack_lenovo_mirage_vr180.py 19700101-001230538.vr.jpg

"""

import os
import sys

if __name__ == "__main__":
  input_filename = sys.argv[1]
  print(input_filename)

exif_cmd = "exiftool -b -xmp:ImageData " + input_filename + " > right.jpg"
print(exif_cmd)
os.system(exif_cmd)


concat_cmd = "convert " + input_filename + " right.jpg +append vr180.jpg"
print(concat_cmd)
os.system(concat_cmd)
