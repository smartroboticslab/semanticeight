#include <iostream>
#include <string>
#include <vector>

#include <se/str_utils.hpp>



int main(int argc, char** argv) {
  const std::string str ("this is supereight");
  const std::vector<std::string> words = split_string(str, ' ');

  std::cout << str << "\n\n";
  for (const auto& word : words) {
    std::cout << word << "\n";
  }
}

