//
//  main.cpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2017/12/13.
//  Copyright © 2017年 柴田 智喜. All rights reserved.
//

#include <iostream>
#include <vector>
#include <Eigen/Dense>
using namespace std;


class Node{
public:
    Node* parent_node;
    Node* more_node;
    Node* less_node;

    double split_value;
    double more_value;
    double less_value;

    Node(){
        parent_node = nullptr;
        more_node = nullptr;
        less_node = nullptr;
    };
    
};

class Tree{
    
//private:
public:
    Node* root_node;
    Node* temp_node;
    vector<Node*> leaf_list;
    
public:
    Tree(){
        set_root_node();
        leaf_list.push_back(root_node);
    }
    
    ~Tree(){
        if (root_node != nullptr){
            del_all_nodes();
        }
    }
    
    Node* get_node(){
            return temp_node;
    }
    
    void set_root_node(){
        root_node = new Node();
        temp_node = root_node;
    }
    
    bool is_parent_node(){
        return temp_node->parent_node != nullptr? true: false;
    }
    
    bool is_more_node(){
        return temp_node->more_node != nullptr? true: false;
    }
    
    bool is_less_node(){
        return temp_node->less_node != nullptr? true: false;
    }
    
    void add_more_node(){
        if(!is_more_node()){
            temp_node->more_node = new Node();
            temp_node->more_node->parent_node = temp_node;
            
            leaf_list.push_back(temp_node->more_node);
            leaf_list_refresh();
        }else{
            cout<<"more_node is not null";
        }
    }
    
    void add_less_node(){
        if(!is_less_node()){
            temp_node->less_node = new Node();
            temp_node->less_node->parent_node = temp_node;
            
            leaf_list.push_back(temp_node->less_node);
            leaf_list_refresh();
        }else{
            cout<<"less_node is not null"<<endl;
        }
    }
    
    void del_this_node(){
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
    
    void to_more_node(){
        if(is_more_node()){
            temp_node = temp_node->more_node;
        }else{
            cout<<"more_node is null"<<endl;
        }
        
    }
    
    void to_less_node(){
            if(is_less_node()){
                temp_node = temp_node->less_node;
            }else{
                cout<<"less_node is null"<<endl;
            }
    }
    
    void to_parent_node(){
        if(is_parent_node()){
            temp_node = temp_node->parent_node;
        }else{
            cout<<"parent_node is null"<<endl;
        }
    }
    
    void to_root_node(){
        if(root_node != nullptr){
            temp_node = root_node;
        }else{
            cout<<"root_node is null"<<endl;
        }
    }
    
    void to_leaf_node(){
        // 子ノードがなくなるまで、再帰的実行
        if(is_more_node()){
            to_more_node();
            to_leaf_node();
        }else if(is_less_node()){
            to_less_node();
            to_leaf_node();
        }
        
    }
    
    void del_all_nodes(){
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
        
        leaf_list.erase(leaf_list.begin(),leaf_list.end());
        leaf_list.shrink_to_fit();
        
    }
    
    void leaf_list_refresh(){
        for(auto iter = leaf_list.begin(); iter != leaf_list.end(); ++iter){
            if(((*iter)->more_node!=nullptr)||((*iter)->less_node!=nullptr)){
                leaf_list.erase(iter);
            }
        }
    }
    
    vector<Node*> get_leaf_list(){
        return leaf_list;
    }
    
};

struct Param{
    int min_samples_leaf;
    int max_depth;
    Param(){
        min_samples_leaf = NULL;
        max_depth = NULL;
    }
};

class CLF_tree{
public:
    CLF_tree(Param inp_param){};
    CLF_tree* fit(MatrixXd X_train,int[], y_train){
        
        
        
        
        return this;
        
        
    };
    int predict(){return 0;};
    bool stop_condition(){return true;};
    Param param;
};

int main(int argc, const char * argv[]) {
    // insert code here...
    cout << "Hello, World!\n";
    Tree tree;
    tree.get_node()->split_value = 1;
    cout<<tree.get_leaf_list().size()<<"@1"<<endl;
    tree.add_more_node();
    tree.to_more_node();
    tree.get_node()->split_value = 2;
    tree.to_parent_node();
    cout<<tree.get_leaf_list().size()<<"@2"<<endl;
    tree.add_less_node();
    tree.to_less_node();
    tree.get_node()->split_value = 3;
    tree.to_parent_node();
    cout<<tree.get_leaf_list().size()<<"@3"<<endl;
    
    tree.to_more_node();
    
    tree.add_more_node();
    tree.to_more_node();
    tree.get_node()->split_value = 4;
    tree.to_parent_node();
    cout<<tree.get_leaf_list().size()<<"@4"<<endl;
    tree.add_less_node();
    tree.to_less_node();
    tree.get_node()->split_value = 5;
    tree.to_parent_node();
    cout<<tree.get_leaf_list().size()<<"@5"<<endl;
    
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
    cout<<tree.get_leaf_list().size()<<"@6"<<endl;
    vector<Node*>leaf_list = tree.get_leaf_list();
    for(auto iter = leaf_list.begin(); iter != leaf_list.end(); ++iter){
        cout<<(*iter)->split_value<<endl;
    }
    
    tree.del_all_nodes();
    cout<<tree.get_leaf_list().size()<<"@7"<<endl;
    //cout<<"root_node"<<tree.root_node<<endl;
    return 0;
}

