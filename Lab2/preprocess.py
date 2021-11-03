MIN_PER_24_HOURS = 60 * 24
MIN_PER_12_HOURS = 60 * 12
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
            for time in range(MIN_PER_12_HOURS):
                strings = fin.readline().split()
                degrees = map(lambda s: float(s), strings)
                degrees = map(process_degree, degrees)
                lohs = map(low_or_high, degrees)
                strings = map(loh_to_str, lohs)
                for patient, loh in enumerate(strings):
                    fout.write(f"{patient + 1}, {time}, {loh}\n")


def stdv(ward):
    with open(f"{ward}.txt", "r") as fin:
        count = 0
        sum_degrees = 0
        sum_squared_degrees = 0
        for time in range(MIN_PER_24_HOURS):
            strings = fin.readline().split()
            degrees = map(lambda s: float(s), strings)
            degrees = map(process_degree, degrees)
            valid_degrees = filter(lambda d: d != INVALID_DEGREE, degrees)
            for degree in valid_degrees:
                sum_degrees += degree
                sum_squared_degrees += degree ** 2
                count += 1
        average = sum_degrees / count
        variance = sum_squared_degrees / count - average ** 2
        return variance ** 0.5


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


if __name__ == "__main__":
    WARDS = 3
    convert2arff(WARDS)
    for i in range(WARDS):
        print("stdv of ward", i + 1, "is", stdv(i + 1))
