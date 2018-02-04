//
//  tree.cpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/01/28.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#include "tree.hpp"

Node::Node(){
    parent_node = nullptr;
    more_node = nullptr;
    less_node = nullptr;
};

Tree::Tree(){
    set_root_node();
}

Tree::~Tree(){
    if (root_node != nullptr){
        del_all_nodes();
    }
}

Node* Tree::get_node(){
    return temp_node;
}

void Tree::set_root_node(){
    root_node = new Node();
    temp_node = root_node;
    temp_node->depth = 0;
    temp_node->path+=string("/");
    
    leaf_list.push_back(root_node);
    node_list.push_back(root_node);
}

bool Tree::is_parent_node(){
    return temp_node->parent_node != nullptr? true: false;
}

bool Tree::is_more_node(){
    return temp_node->more_node != nullptr? true: false;
}

bool Tree::is_less_node(){
    return temp_node->less_node != nullptr? true: false;
}

void Tree::add_more_node(){
    if(!is_more_node()){
        // 初期化
        temp_node->more_node = new Node();
        // parent_nodeにtemp_nodeを設定
        temp_node->more_node->parent_node = temp_node;
        // depth設定
        temp_node->more_node->depth = temp_node->depth + 1;
        
        leaf_list.push_back(temp_node->more_node);
        leaf_list_refresh();
        node_list.push_back(temp_node->more_node);
    }else{
        cout<<"more_node is not null";
    }
}

void Tree::add_less_node(){
    if(!is_less_node()){
        // 初期化
        temp_node->less_node = new Node();
        // parent_nodeにtemp_nodeを設定
        temp_node->less_node->parent_node = temp_node;
        // depth設定
        temp_node->less_node->depth = temp_node->depth + 1;
        
        leaf_list.push_back(temp_node->less_node);
        leaf_list_refresh();
        node_list.push_back(temp_node->more_node);
    }else{
        cout<<"less_node is not null"<<endl;
    }
}

void Tree::del_this_node(){
    Node* del_node = temp_node;
    to_parent_node();
    
    if(del_node == temp_node->more_node){
        temp_node->more_node = nullptr;
        delete del_node;
    }else if(del_node == temp_node->less_node){
        temp_node->less_node = nullptr;
        delete del_node;
    }else{
        cout<<"del node address error"<<endl;
    }
    
}

void Tree::to_more_node(){
    if(is_more_node()){
        temp_node = temp_node->more_node;
    }else{
        cout<<"more_node is null"<<endl;
    }
    
}

void Tree::to_less_node(){
    if(is_less_node()){
        temp_node = temp_node->less_node;
    }else{
        cout<<"less_node is null"<<endl;
    }
}

void Tree::to_parent_node(){
    if(is_parent_node()){
        temp_node = temp_node->parent_node;
    }else{
        cout<<"parent_node is null"<<endl;
    }
}

void Tree::to_root_node(){
    if(root_node != nullptr){
        temp_node = root_node;
    }else{
        cout<<"root_node is null"<<endl;
    }
}

void Tree::to_leaf_node(){
    // 子ノードがなくなるまで、再帰的実行
    if(is_more_node()){
        to_more_node();
        to_leaf_node();
    }else if(is_less_node()){
        to_less_node();
        to_leaf_node();
    }
    
}

void Tree::del_all_nodes(){
    
    //はじめに葉ノードに移動
    to_leaf_node();
    //cout<<"hello"<<endl;
    while(is_parent_node()){//次の処理を対象がルートになるまで（対象がルートかつ、葉ノードである）繰り返す。
        // 現在のノードを消して、親ノードに移動
        del_this_node();
        // 葉ノードまで移動
        to_leaf_node();
        
    }
    
    // ルートノードの削除
    delete root_node;
    root_node = nullptr;
    
    //ノードのリストの初期化
    leaf_list.erase(leaf_list.begin(),leaf_list.end());
    leaf_list.shrink_to_fit();
    node_list.erase(node_list.begin(),node_list.end());
    node_list.shrink_to_fit();
    
}

void Tree::leaf_list_refresh(){
    for(auto iter = leaf_list.begin(); iter != leaf_list.end(); ++iter){
        if(((*iter)->more_node!=nullptr)||((*iter)->less_node!=nullptr)){
            leaf_list.erase(iter);
        }
    }
}

vector<Node*> Tree::get_leaf_list(){
    return leaf_list;
}
