NOON_12_O_CLOCK = 60 * 12
LOW = 0
HIGH = 1
HIGH_THRESHOLD = 37
MIN_VALID_CELSIUS = 36
MAX_VALID_CELSIUS = 43
INVALID_DEGREE = None
INVALID_DEGREE_STR = "?"


def convert2arff(num_of_files):
    for ward in range(1, num_of_files + 1):
        with open(f"{ward}.txt", "r") as fin, open(f"temp{ward}.arff", "w") as fout:
            fout.write(f"@relation ward{ward}_patients_temperatures\n")
            fout.write("@attribute patients_ID numeric\n")
            fout.write("@attribute time numeric\n")
            fout.write("@attribute temperatue {0, 1}\n\n")
            fout.write("@data\n")
            for time in range(NOON_12_O_CLOCK):
                strings = fin.readline().split()
                str_to_float = lambda s: float(s)
                degrees = map(str_to_float, strings)
                degrees = map(process_degree, degrees)
                lohs = map(low_or_high, degrees)
                strings = map(loh_to_str, lohs)
                for patient, loh in enumerate(strings):
                    fout.write(f"{patient + 1}, {time}, {loh}\n")


def process_degree(degree):
    if degree < MIN_VALID_CELSIUS:
        return INVALID_DEGREE
    if degree > MAX_VALID_CELSIUS:
        celsius = (degree - 32) * 5 / 9
        if MIN_VALID_CELSIUS <= celsius <= MAX_VALID_CELSIUS:
            return celsius
        return INVALID_DEGREE
    return degree


def low_or_high(celsius):
    if celsius == INVALID_DEGREE:
        return INVALID_DEGREE
    if celsius > HIGH_THRESHOLD:
        return HIGH
    return LOW


def loh_to_str(degree):
    if degree == INVALID_DEGREE:
        return INVALID_DEGREE_STR
    return str(degree)


convert2arff(3)
