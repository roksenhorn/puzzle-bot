/*
 * Usage: ./find_islands <input_directory> <output_directory> <min_island_area>
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pthread.h>

#define MAX_THREADS 14
#define max(a, b) ((a) > (b) ? (a) : (b))

/*
 * Island finding
 */

typedef struct {
    int rows;
    int cols;
    int origin_x;
    int origin_y;
    int **matrix;
} Island;

int min_island_area = 0;

void mark_island(int **grid, int rows, int cols, int x, int y, int **visited, int *min_x, int *max_x, int *min_y, int *max_y, int *area, int island_id) {
    typedef struct {
        int x;
        int y;
    } Point;

    int stack_size = rows * cols;
    Point *stack = (Point *)malloc(stack_size * sizeof(Point));

    int top = -1;
    stack[++top] = (Point){x, y};

    while (top >= 0) {
        Point p = stack[top--];
        int px = p.x;
        int py = p.y;

        if (px < 0 || px >= rows || py < 0 || py >= cols || grid[px][py] == 0 || visited[px][py]) {
            continue;
        }

        visited[px][py] = island_id;
        (*area)++;
        if (px < *min_x) *min_x = px;
        if (px > *max_x) *max_x = px;
        if (py < *min_y) *min_y = py;
        if (py > *max_y) *max_y = py;

        // Push all 4 directions onto the stack
        stack[++top] = (Point){px + 1, py};
        stack[++top] = (Point){px - 1, py};
        stack[++top] = (Point){px, py + 1};
        stack[++top] = (Point){px, py - 1};
    }

    free(stack);
}

Island *create_island(int min_x, int max_x, int min_y, int max_y) {
    int padding = 1;
    int rows = max_x - min_x + 1 + (2 * padding);
    int cols = max_y - min_y + 1 + (2 * padding);
    Island *island = (Island *)malloc(sizeof(Island));
    island->rows = rows;
    island->cols = cols;
    island->origin_x = min_x - padding;
    island->origin_y = min_y - padding;
    island->matrix = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        island->matrix[i] = (int *)calloc(cols, sizeof(int));
    }
    return island;
}

Island **find_islands(int **grid, int rows, int cols, int min_island_area, int ignore_islands_along_border, int *num_islands) {
    int **visited = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        visited[i] = (int *)calloc(cols, sizeof(int));
    }

    Island **islands = NULL;
    *num_islands = 0;

    int island_id = 1; // Unique identifier for each island
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] == 1 && visited[i][j] == 0) {
                int area = 0;
                int min_x = i, max_x = i, min_y = j, max_y = j;
                mark_island(grid, rows, cols, i, j, visited, &min_x, &max_x, &min_y, &max_y, &area, island_id);

                int on_border = (min_x == 0 || max_x == rows - 1 || min_y == 0 || max_y == cols - 1);

                if (area >= min_island_area && (!ignore_islands_along_border || !on_border)) {
                    Island *new_island = create_island(min_x, max_x, min_y, max_y);
                    for (int x = min_x; x <= max_x; x++) {
                        for (int y = min_y; y <= max_y; y++) {
                            if (visited[x][y] == island_id) {
                                new_island->matrix[x - min_x + 1][y - min_y + 1] = 1;
                            }
                        }
                    }
                    islands = (Island **)realloc(islands, (*num_islands + 1) * sizeof(Island *));
                    islands[*num_islands] = new_island;
                    (*num_islands)++;
                }
                island_id++; // Increment the unique identifier for the next island
            }
        }
    }

    for (int i = 0; i < rows; i++) {
        free(visited[i]);
    }
    free(visited);

    return islands;
}

/*
 * Cleaning
 */

int _is_straggler(int **mat, int x, int y) {
    int neighbors = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            neighbors += mat[x + dx][y + dy];
        }
    }
    return neighbors <= 2;
}

void remove_stragglers(int **matrix, int rows, int cols) {
    // clean up any tiny stragglers where the island is only connected by a single point
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            if (matrix[i][j] == 1 && _is_straggler(matrix, i, j)) {
                // any time we find a point that is dangling, remove it
                matrix[i][j] = 0;
                // removing this point might have made a prior point into another straggler
                // so we backtrack and check this area again
                i = max(1, i - 2);
                j = max(1, j - 2);
            }
        }
    }
}

/*
 * BMP utils
 */

#pragma pack(push, 1)
typedef struct {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BITMAPFILEHEADER;

typedef struct {
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

int **load_binary_bitmap(const char *filename, int *width, int *height) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error: Unable to open file %s\n", filename);
        return NULL;
    }

    BITMAPFILEHEADER fileHeader;
    if (fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, file) != 1) {
        printf("Error: Failed to read file header\n");
        fclose(file);
        return NULL;
    }

    if (fileHeader.bfType != 0x4D42) {
        printf("Error: Not a BMP file\n");
        fclose(file);
        return NULL;
    }

    BITMAPINFOHEADER infoHeader;
    if (fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, file) != 1) {
        printf("Error: Failed to read info header\n");
        fclose(file);
        return NULL;
    }

    *width = infoHeader.biWidth;
    *height = infoHeader.biHeight;

    int **grid = (int **)malloc(*height * sizeof(int *));

    for (int i = 0; i < *height; i++) {
        grid[i] = (int *)malloc(*width * sizeof(int));
    }

    fseek(file, fileHeader.bfOffBits, SEEK_SET);

    int row_padded = ((*width + 31) / 32) * 4;
    unsigned char *row_data = (unsigned char *)malloc(row_padded);

    for (int i = 0; i < *height; i++) {
        if (fread(row_data, sizeof(unsigned char), row_padded, file) != row_padded) {
            printf("Error: Failed to read bitmap data\n");
            free(row_data);
            for (int j = 0; j < *height; j++) {
                free(grid[j]);
            }
            free(grid);
            fclose(file);
            return NULL;
        }
        for (int j = 0; j < *width; j++) {
            int byte_index = j / 8;
            int bit_index = 7 - (j % 8);
            int pixel_value = (row_data[byte_index] >> bit_index) & 1;
            grid[*height - i - 1][j] = pixel_value;
        }
    }

    free(row_data);
    fclose(file);
    return grid;
}

// Function to save an island as a BMP file
void save_island_as_bmp(Island *island, const char *filename) {
    int width = island->cols;
    int height = island->rows;
    int row_padded = (width + 31) / 32 * 4;
    int filesize = 54 + row_padded * height;

    unsigned char *img = (unsigned char *)calloc(row_padded * height, sizeof(unsigned char));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int bit_index = 7 - (j % 8);
            if (island->matrix[height - 1 - i][j] == 1) {
                img[i * row_padded + j / 8] |= (1 << bit_index);
            }
        }
    }

    BITMAPFILEHEADER fileHeader;
    fileHeader.bfType = 0x4D42;
    fileHeader.bfSize = filesize;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = 54;

    BITMAPINFOHEADER infoHeader;
    infoHeader.biSize = 40;
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 1;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = row_padded * height;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 2;
    infoHeader.biClrImportant = 2;

    unsigned char bmp_pad[3] = {0, 0, 0};
    unsigned char bmp_color_table[8] = {
        0, 0, 0, 0, 255, 255, 255, 0};

    FILE *file = fopen(filename, "wb");
    fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, file);
    fwrite(&infoHeader, sizeof(BITMAPINFOHEADER), 1, file);
    fwrite(bmp_color_table, sizeof(bmp_color_table), 1, file);
    fwrite(img, row_padded * height, 1, file);

    fclose(file);
    free(img);
}

void free_islands(Island **islands, int num_islands) {
    for (int i = 0; i < num_islands; i++) {
        Island *island = islands[i];
        for (int j = 0; j < island->rows; j++) {
            free(island->matrix[j]);
        }
        free(island->matrix);
        free(island);
    }
    free(islands);
}

/*
 * Processing
 */

void extract(const char *filepath, const char *filename, const char *output_directory_path) {
    printf("Extracting from %s\n", filepath);

    int width, height;
    int **grid = load_binary_bitmap(filepath, &width, &height);
    if (grid == NULL) {
        return;
    }

    // first preprocess the image and remove any stragglers
    remove_stragglers(grid, height, width);

    // now extract all large islands that don't touch the border
    int ignore_islands_along_border = 1;
    int num_islands;
    Island **islands = find_islands(grid, height, width, min_island_area, ignore_islands_along_border, &num_islands);

    // Save each island as a BMP file
    for (int i = 0; i < num_islands; i++) {
        // remove ".bmp" from the end of the filename and add the island's origin coordinates
        char trimmed_filename[256];
        strcpy(trimmed_filename, filename);
        trimmed_filename[strlen(filename) - 4] = '\0';

        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s/%s_(%d,%d).bmp", output_directory_path, trimmed_filename, islands[i]->origin_y, islands[i]->origin_x);
        save_island_as_bmp(islands[i], output_filename);
    }

    free_islands(islands, num_islands);

    for (int i = 0; i < height; i++) {
        free(grid[i]);
    }
    free(grid);
}

typedef struct {
    char filepath[1024];
    char filename[1024];
} Task;

Task task_queue[5000];
int task_count = 0;
char *output_directory_path;
pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t task_cond = PTHREAD_COND_INITIALIZER;

void* worker(void* arg) {
    while (1) {
        // grab the next task
        pthread_mutex_lock(&task_mutex);
        if (task_count == 0) {
            // if there are no tasks left, we're done!
            pthread_mutex_unlock(&task_mutex);
            break;
        }
        Task task = task_queue[--task_count];
        pthread_mutex_unlock(&task_mutex);

        extract(task.filepath, task.filename, output_directory_path);
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_directory> <output_directory> <min_island_area>\n", argv[0]);
        return 1;
    }

    min_island_area = atoi(argv[3]);

    char *input_directory_path = argv[1];
    DIR *dir = opendir(input_directory_path);
    if (!dir) {
        fprintf(stderr, "Could not open directory: %s\n", input_directory_path);
        return 1;
    }

    output_directory_path = argv[2];
    DIR *output_dir = opendir(output_directory_path);
    if (!output_dir) {
        fprintf(stderr, "Could not open directory: %s\n", output_directory_path);
        return 1;
    }
    closedir(output_dir);
    struct stat st = {0};

    // create a list of tasks, one per file
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            char *filename = entry->d_name;
            if (strstr(filename, ".bmp")) {
                char filepath[1024];
                snprintf(filepath, sizeof(filepath), "%s/%s", input_directory_path, filename);

                Task task;
                strcpy(task.filepath, filepath);
                strcpy(task.filename, filename);
                task_queue[task_count] = task;

                task_count++;
            }
        }
    }

    pthread_t threads[MAX_THREADS];
    for (int i = 0; i < MAX_THREADS; ++i) {
        pthread_create(&threads[i], NULL, worker, NULL);
    }
    for (int i = 0; i < MAX_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }
    closedir(dir);

    return 0;
}
