/*****************************************************************
    Some helper macros for benchmarking and CUDA functions

    Copyright (C) 2013 Stefan Grönroos

    Authors: Stefan Grönroos <stefan.gronroos@abo.fi>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

********************************************************************/

#include <cutil.h>

#ifdef BENCHMARKING
/* Simple benchmarking macros. See also helpers.c
 * Use START_CLOCK(x) to start timer 0-4
 * Use BENCHMARK_NOW(x, "String") to output a benchmark message telling the time difference between
 * calling START_CLOCK and now. x is here also a timer ID (0-4)
 */

extern struct timespec time1[5], time2;
#define CLOCK_TYPE CLOCK_REALTIME /* CLOCK_REALTIME; CLOCK_PROCESS_CPUTIME_ID */
#define START_CLOCK(id) extern struct timespec time1[5], time2; clock_gettime(CLOCK_TYPE, &time1[id]);
#define BENCHMARK_NOW(id, string) clock_gettime(CLOCK_TYPE, &time2); \
                                printf("\nBenchmark: " string ": %i:%i (%Lf)\n", diff(time1[id],time2).tv_sec, diff(time1[id],time2).tv_nsec, \
                                        (long double)diff(time1[id],time2).tv_sec + (long double)diff(time1[id],time2).tv_nsec/1000000000.0);
struct timespec diff(struct timespec start, struct timespec end);
#else
/* empty macros if benchmarking not enabled */
#define START_CLOCK(id)
#define BENCHMARK_NOW(id, string)
#endif
