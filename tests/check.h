#ifndef CANTATAS_TEST_CHECK_H
#define CANTATAS_TEST_CHECK_H

#include <cstdlib>
#include <iostream>

#define CHECK(cond)                                                                                \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond << std::endl;          \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

#define CHECK_EQ(a, b)                                                                             \
  do {                                                                                             \
    auto _va = (a);                                                                                \
    auto _vb = (b);                                                                                \
    if (!(_va == _vb)) {                                                                           \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK_EQ failed: " #a " = " << _va            \
                << ", " #b " = " << _vb << std::endl;                                              \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

#endif
