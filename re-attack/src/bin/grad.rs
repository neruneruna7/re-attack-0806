// 勾配計算について簡単に実装してみる

use std::cell::RefCell;
use std::rc::Rc;

type NodeRef = Rc<RefCell<Node>>;
struct Node {
    value: f64,
    grad: f64,
    parents: Vec<NodeRef>,
    backward: Option<Box<dyn Fn(f64)>>,
}
impl Node {
    fn new(value: f64) -> NodeRef {
        Rc::new(RefCell::new(Node {
            value,
            grad: 0.0,
            parents: vec![],
            backward: None,
        }))
    }
}

fn add(a: &NodeRef, b: &NodeRef) -> NodeRef {
    let out = Node::new(a.borrow().value + b.borrow().value);
    out.borrow_mut().parents = vec![Rc::clone(a), Rc::clone(b)];
    let a_clone = Rc::clone(a);
    let b_clone = Rc::clone(b);

    out.borrow_mut().backward = Some(Box::new(move |grad_out| {
        a_clone.borrow_mut().grad += grad_out;
        b_clone.borrow_mut().grad += grad_out;
    }));
    out
}

fn mul(a: &NodeRef, b: &NodeRef) -> NodeRef {
    let out = Node::new(a.borrow().value * b.borrow().value);
    out.borrow_mut().parents = vec![Rc::clone(a), Rc::clone(b)];
    let a_clone = Rc::clone(a);
    let b_clone = Rc::clone(b);
    let a_val = a.borrow().value;
    let b_val = b.borrow().value;

    out.borrow_mut().backward = Some(Box::new(move |grad_out| {
        a_clone.borrow_mut().grad += b_val * grad_out;
        b_clone.borrow_mut().grad += a_val * grad_out;
    }));
    out
}

fn topo_sort(node: &NodeRef, visited: &mut Vec<NodeRef>, order: &mut Vec<NodeRef>) {
    if visited.iter().any(|n| Rc::ptr_eq(n, node)) {
        return;
    }
    visited.push(Rc::clone(node));
    for p in &node.borrow().parents {
        topo_sort(p, visited, order);
    }
    order.push(Rc::clone(node));
}

fn backward(node: &NodeRef) {
    let mut visited = vec![];
    let mut order = vec![];
    topo_sort(node, &mut visited, &mut order);

    node.borrow_mut().grad = 1.0;

    // 逆順に処理
    while let Some(n) = order.pop() {
        if let Some(ref backward_fn) = n.borrow().backward {
            let grad_val = n.borrow().grad;
            backward_fn(grad_val);
        }
    }
}

fn main() {
    // x=2, y=3 のとき z = x * y + y
    let x = Node::new(2.0);
    let y = Node::new(3.0);
    let z = add(&mul(&x, &y), &y);

    backward(&z);

    println!("z value = {}", z.borrow().value); // 9
    println!("dz/dx = {}", x.borrow().grad); // 3
    println!("dz/dy = {}", y.borrow().grad); // 3 + 2 = 5
}
