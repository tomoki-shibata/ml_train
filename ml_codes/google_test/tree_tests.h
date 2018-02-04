//
//  tree_tests.h
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/01/13.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#ifndef tree_tests_h
#define tree_tests_h

#include "gtest/gtest.h"
#include "tree.hpp"

namespace
{
    class TreeTest : public ::testing::Test
    {
        
    protected:
        TreeTest()
        {
            
        }
        
        virtual ~TreeTest()
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

TEST_F(TreeTest, basic_tree_test)
{
    EXPECT_EQ(1, something(1));
    Tree tree;
    tree.get_node()->split_value = 1;
    EXPECT_EQ(1, tree.get_leaf_list().size());
    EXPECT_EQ(0,tree.get_node()->depth);
    tree.add_more_node();
    tree.to_more_node();
    tree.get_node()->split_value = 2;
    tree.to_parent_node();
    EXPECT_EQ(1, tree.get_leaf_list().size());
    tree.add_less_node();
    tree.to_less_node();
    tree.get_node()->split_value = 3;
    tree.to_parent_node();
    EXPECT_EQ(2, tree.get_leaf_list().size());
    
    tree.to_more_node();
    
    tree.add_more_node();
    tree.to_more_node();
    tree.get_node()->split_value = 4;
    tree.to_parent_node();
    EXPECT_EQ(2, tree.get_leaf_list().size());
    tree.add_less_node();
    tree.to_less_node();
    tree.get_node()->split_value = 5;
    tree.to_parent_node();
    EXPECT_EQ(3, tree.get_leaf_list().size());
    
    tree.to_parent_node();
    //tree.to_parent_node();
    
    tree.to_less_node();
    tree.add_less_node();
    tree.to_less_node();
    tree.get_node()->split_value=11;
    tree.to_parent_node();
    tree.add_more_node();
    tree.to_more_node();
    tree.get_node()->split_value=12;
    tree.to_parent_node();
    
    tree.to_more_node();
    
    tree.add_less_node();
    tree.to_less_node();
    tree.get_node()->split_value=13;
    tree.to_parent_node();
    
    //tree.add_more_node();
    //cout<<tree.get_leaf_list().size()<<endl;
    //tree.del_this_node();
    //tree.del_this_node();
    EXPECT_EQ(4, tree.get_leaf_list().size());
    vector<Node*>leaf_list = tree.get_leaf_list();
    for(auto iter = leaf_list.begin(); iter != leaf_list.end(); ++iter){
        //cout<<(*iter)->split_value<<endl;
    }
    
    tree.del_all_nodes();
    //cout<<tree.get_leaf_list().size()<<"@7"<<endl;
    EXPECT_EQ(0, tree.get_leaf_list().size());
}

TEST_F(TreeTest, tree_depth_test)
{
    EXPECT_EQ(1, something(1));
    Tree tree;
    tree.get_node()->split_value = 1;
    EXPECT_EQ(0,tree.get_node()->depth);
    tree.add_more_node();
    tree.to_more_node();
    EXPECT_EQ(1,tree.get_node()->depth);
    tree.get_node()->split_value = 2;
    tree.to_parent_node();
    EXPECT_EQ(0,tree.get_node()->depth);
    tree.add_less_node();
    tree.to_less_node();
    EXPECT_EQ(1,tree.get_node()->depth);
    tree.get_node()->split_value = 3;
    tree.to_parent_node();
    EXPECT_EQ(2, tree.get_leaf_list().size());
    
    tree.to_more_node();
    
    tree.add_more_node();
    tree.to_more_node();
    EXPECT_EQ(2,tree.get_node()->depth);
    tree.get_node()->split_value = 4;
    tree.to_parent_node();
    EXPECT_EQ(1,tree.get_node()->depth);
    tree.add_less_node();
    tree.to_less_node();
    EXPECT_EQ(2,tree.get_node()->depth);
    tree.get_node()->split_value = 5;
    tree.to_parent_node();
    EXPECT_EQ(1,tree.get_node()->depth);
    
    tree.to_parent_node();
    //tree.to_parent_node();
    
    tree.to_less_node();
    tree.add_less_node();
    tree.to_less_node();
    EXPECT_EQ(2,tree.get_node()->depth);
    tree.get_node()->split_value=11;
    tree.to_parent_node();
    tree.add_more_node();
    tree.to_more_node();
    EXPECT_EQ(2,tree.get_node()->depth);
    tree.get_node()->split_value=12;
    tree.to_parent_node();
    
    tree.to_more_node();
    
    tree.add_less_node();
    tree.to_less_node();
    EXPECT_EQ(3,tree.get_node()->depth);
    tree.get_node()->split_value=13;
    tree.to_parent_node();
    
    tree.to_less_node();
    
    tree.add_less_node();
    tree.to_less_node();
    EXPECT_EQ(4,tree.get_node()->depth);
    
    tree.add_less_node();
    tree.to_less_node();
    EXPECT_EQ(5,tree.get_node()->depth);
    
    tree.add_less_node();
    tree.to_less_node();
    EXPECT_EQ(6,tree.get_node()->depth);
    
    //tree.add_more_node();
    //cout<<tree.get_leaf_list().size()<<endl;
    //tree.del_this_node();
    //tree.del_this_node();
    vector<Node*>leaf_list = tree.get_leaf_list();
    for(auto iter = leaf_list.begin(); iter != leaf_list.end(); ++iter){
        //cout<<(*iter)->split_value<<endl;
    }
    
    tree.del_all_nodes();
    //cout<<tree.get_leaf_list().size()<<"@7"<<endl;
    EXPECT_EQ(0,tree.get_node()->depth);
}




#endif /* tree_tests_h */
