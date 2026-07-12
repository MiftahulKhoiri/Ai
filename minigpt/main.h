// main.h
#pragma once
#include <string>

// Helper: cek apakah file pada "path" ada di disk.
// Dipakai main.cpp untuk mengecek keberadaan checkpoint sebelum load.
bool file_exists(const std::string& path);