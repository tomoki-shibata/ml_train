//
//  main.cpp
//  google_test
//
//  Created by 柴田 智喜 on 2018/01/08.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//
/*
#include <iostream>

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    return 0;
}
*/

#define GTEST_HAS_TR1_TUPLE 0
//#include "tree_tests.h"
//#include "clf_tree_test.h"
#include "random_forest_test.h"
#include <random>
#include <iostream>
#include "gtest/gtest.h"
using namespace std;
int main(int argc, const char * argv[])
{

    ::testing::InitGoogleTest(&argc, (char **)argv);
    return RUN_ALL_TESTS();
}
