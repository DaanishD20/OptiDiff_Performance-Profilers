#include <stdio.h> 

int square(int x) { 
    return x * x;
}

int main() { 
    int n = 5;
    int result = square(n);
    printf("Result: %d\n", result);
    return 0;
}
