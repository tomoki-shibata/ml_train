//
//  main.cpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2017/12/13.
//  Copyright © 2017年 柴田 智喜. All rights reserved.
//

//#include "tree.h"
#include "clf_tree.hpp"
//#include "trial.hpp"
//#include "clf_tree.hpp"
using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "Hello, World!\n";
    
    //cout<<one()<<endl;
    
    //test_calss test_print;
    //test_print.print();
    
    CLF_tree::CLF_tree_param param;
    //cout<<"param.max_depth->"<<param.max_depth<<endl;
    
    CLF_tree clf_tree(param);
    //CLF_tree clf_tree;
    //cout<<clf_tree.test<<endl;
    
    Tree t;
    cout<<t.get_node()->depth<<endl;
    cout<<"end"<<endl;
    
    
    //trial tr;
    //tr.print();
    //tr.print_2();
    
    
    
    return 0;
}

