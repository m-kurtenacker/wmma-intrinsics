#include <stdio.h>

// Writes each float in the array to "test.csv", prefixed by the given string.
// If "test.csv" already exists, new lines will be appended at the end.
// Each line format: <prefix>,<float_value>
void write_csv(const char *prefix, const float *values, size_t count) {
    FILE *fp = fopen("test.csv", "a"); // "a" mode for append
    if (!fp) {
        perror("Failed to open test.csv");
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        fprintf(fp, "%s,%.6f\n", prefix, values[i]);
    }
    fclose(fp);
}
