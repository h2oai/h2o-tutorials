## This script is applied to the covtype dataset available at https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/covtype/covtype.data
## It converts expanded categorical levels back to dense columns

f = open("covtype.data","r")
lines = f.readlines()
data = [l.split(",") for l in lines]
def which_level(x, f, t):
  for i in range(f,t):
    if x[i] == '1':
      return i - f
      raise Exception()
cats1 = [which_level(l,10,14) for l in data]
cats2 = [which_level(l,14,54) for l in data]
data2 = [data[i][0:10] + ['area_' + str(cats1[i]),'type_' + str(cats2[i]), 'class_' + str(data[i][54])] for i in range(len(data))]
output_data = [','.join(x) for x in data2]
f = open('covtype_edited.csv','w')
f.write("Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area, Soil_Type, Cover_Type\n")
f.write("".join(output_data) )
f.close()
