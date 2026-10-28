/**
 * @file xgb_glue.c
 * @brief Compiles the RCRE implementation exactly once for this project.
 *
 * The amalgamated dist header (scripts/amalgamate.py output) contains all
 * RCRE sources under #ifdef RCRE_IMPLEMENTATION. RCRE_DIR is injected by
 * the Makefile (default: sibling ../rcre/dist).
 */
#define RCRE_IMPLEMENTATION
#include "rcre.h"
