import io
from PIL import Image, ImageDraw

im = Image.new("L", (512, 512), 0)
d = ImageDraw.Draw(im)
d.ellipse([40, 40, 472, 472], outline=170, width=26)
d.ellipse([66, 62, 446, 450], outline=150, width=10)
d.ellipse([100, 95, 412, 417], fill=95)
d.line([(256, 100), (256, 412)], fill=20, width=4)
d.ellipse([188, 235, 246, 285], fill=30)
d.ellipse([266, 235, 324, 285], fill=30)
buf = io.BytesIO()
im.save(buf, format="PNG")
with open(r"C:\Users\horita\ct-report-assistant\test_ct_512.png", "wb") as f:
    f.write(buf.getvalue())
print("wrote test_ct_512.png", len(buf.getvalue()), "bytes")
