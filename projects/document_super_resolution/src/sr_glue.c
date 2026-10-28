/**
 * @file sr_glue.c
 * @brief Compiles the RCRE single-header implementation exactly once.
 *        Every other TU gets declarations only via #include "rcre.h".
 */
#define RCRE_IMPLEMENTATION
#include "rcre.h"
