class GCode:

    @staticmethod
    def generate_gcode_file(points, file_name):
        code = "G90\n"
        code = code + "G1 Z1 F2400\n"
        
        if (len(points) == 0):
            return code

        code = code + (f'G1 X{points[0][0]} Y{points[0][1]} Z1 F2400\n')
        code = code + (f'G1 Z0 F2400\n')

        for point in points:
            code = code + (f'G1 X{point[0]} Y{point[1]} F2400\n')

        f = open(file_name, "w")
        f.write(code)
        f.close()

        

