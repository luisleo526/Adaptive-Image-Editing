from create_dof import create_dof_data
from create_fog import create_fog_data
import getopt
import sys

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--rgb' and strArgument != '':
        arguments_Rgb = strArgument  # path to the input image
    if strOption == '--depth' and strArgument != '':
        arguments_Depth = strArgument  # path to the depth estimation
    if strOption == '--out' and strArgument != '':
        arguments_Out = strArgument  # path to the output

create_dof_data(arguments_Rgb, arguments_Depth, arguments_Out, 1)
create_fog_data(arguments_Rgb, arguments_Depth, arguments_Out, 1)
