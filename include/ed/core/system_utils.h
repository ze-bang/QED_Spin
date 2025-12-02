#ifndef SYSTEM_UTILS_H
#define SYSTEM_UTILS_H

#include <cstdlib>
#include <string>
#include <iostream>

/**
 * @brief Execute a system command and check for errors
 * @param cmd Command to execute
 * @return true if command succeeded, false otherwise
 */
inline bool safe_system_call(const std::string& cmd) {
    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Warning: System command failed with code " << result << ": " << cmd << std::endl;
        return false;
    }
    return true;
}

#endif // SYSTEM_UTILS_H
