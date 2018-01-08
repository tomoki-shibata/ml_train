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
#include "gtest/gtest.h"
#include "tree.h"

namespace
{
    class MyTest : public ::testing::Test
    {
    protected:
        MyTest()
        {
            
        }
        
        virtual ~MyTest()
        {
            
        }
        
        virtual void SetUp()
        {
            
        }
        
        virtual void TearDown()
        {
            
        }
    };
}

int something(int i)
{
    return i;
}

TEST_F(MyTest, MyTestSuite)
{
    EXPECT_EQ(1, something(1));
}

TEST_F(MyTest, pinnkodaci)
{
    EXPECT_EQ(2, something(2));
}

TEST_F(MyTest, pinnkodaci2)
{
    EXPECT_EQ(1, one());
}


int main(int argc, const char * argv[])
{
    ::testing::InitGoogleTest(&argc, (char **)argv);
    return RUN_ALL_TESTS();
}
