from ocrmac import ocrmac
from PIL import Image
path = "/Users/michaelwu/modelscope/demo/invoice_example/1.png"
img = Image.open(path)
code = (1320,56,1544,96)
date = (1319,105,1478,138)
buyer_name_crop_range = (189,249,498,282)
#buyer_id_crop_range = (469,332,712,363)
seller_name_crop_range = (944,249,1230,287)
seller_id_crop_range = (1232,333,1466,362)
total_money_crop_range = (1221,734,1301,767)
img = img.crop(seller_name_crop_range)
img.save("tmp.png")
import easyocr
reader = easyocr.Reader(['ch_sim'], gpu=False)
text = reader.readtext("tmp.png")
print(text)
#from paddleocr import PaddleOCR, draw_ocr
#ocr = PaddleOCR(use_angle_cls = True, lang = "ch")
#text = ocr.ocr(img.crop(code), cls = True)
#print(text)
crop_ranges = [code, date, buyer_name_crop_range, seller_name_crop_range, seller_id_crop_range, total_money_crop_range]
information = []
information_title = ["开票号码","开票日期","购买方名称", "销售方名称", "销售方统一社会信用代码", "价税合计"]
for crop_range in crop_ranges:
    tmp = ocrmac.OCR(img.crop(crop_range),language_preference=['zh-Hans']).recognize()
    information.append(tmp[0][0])
print(information)
