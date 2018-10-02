from PIL import Image

#we have 48894 pixel data, so we may have a 174*281 graph
color_x = 174
color_y = 281

data_x,data_y  = 361,641

def csv2image(csvfile,x,y):
    im = Image.new("RGB",(x,y))
    with open(csvfile) as f:
        for i in range(x):
            for j in range(y):
                line = f.readline()
                rgb = line.split()[0].split(",")
                im.putpixel((i,j),(int(rgb[0]),int(rgb[1]),int(rgb[2])))
    im.show()


#csv2image("color.csv",color_x,color_y)
csv2image("data.csv",data_x,data_y)
