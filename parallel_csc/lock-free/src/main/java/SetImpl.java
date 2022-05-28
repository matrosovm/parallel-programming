import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicMarkableReference;
import java.util.List;

public class SetImpl<T extends Comparable<T>> implements Set<T> {

    private final Node<T> head;

    private static class Node<K extends Comparable<K>> {
        final K key;
        final AtomicMarkableReference<Node<K>> next;

        Node() {
            key = null;
            next = new AtomicMarkableReference<>(null, false);
        }

        Node(Node<K> nextNode, K value) {
            this.key = value;
            next = new AtomicMarkableReference<>(nextNode, false);
        }
    }

    @SafeVarargs
    SetImpl(T... args) {
        head = new Node<>();
        for (T item : args) {
            add(item);
        }
    }

    private List<Node<T>> findNodesCouple(T value) {
        boolean isBlocking;
        Node<T> prev;
        do {
            isBlocking = false;
            prev = head;
            Node<T> current = prev.next.getReference();
            while (current != null) {
                if (!current.next.isMarked() && current.key.compareTo(value) == 0) {
                    return Arrays.asList(prev, current);
                }
                Node<T> next = current.next.getReference();
                if (current.next.isMarked() &&
                    !prev.next.compareAndSet(current, next, false, false)) {
                    isBlocking = true;
                    break;
                }
                prev = current;
                current = next;
            }
        } while (isBlocking);
        return Arrays.asList(prev, null);
    }

    @Override
    public boolean add(T value) {
        while (true) {
            List<Node<T>> nodesCouple = findNodesCouple(value);
            Node<T> prev = nodesCouple.get(0);
            Node<T> current = nodesCouple.get(1);
            if (current != null && current.key.compareTo(value) == 0) {
                return false;
            }
            if (prev != null && prev.next.compareAndSet(current, new Node<>(current, value),
                false, false))
                return true;
        }
    }

    @Override
    public boolean remove(T value) {
        while (true) {
            List<Node<T>> nodesCouple = findNodesCouple(value);
            Node<T> prev = nodesCouple.get(0);
            Node<T> current = nodesCouple.get(1);
            if (current == null) {
                return false;
            }
            Node<T> next = current.next.getReference();
            if (current.next.attemptMark(next, true)) {
                prev.next.compareAndSet(current, next, false, false);
                return true;
            }
        }
    }

    @Override
    public boolean contains(T value) {
        List<Node<T>> nodesCouple = findNodesCouple(value);
        Node<T> current = nodesCouple.get(1);
        return current != null && !current.next.isMarked();
    }

    @Override
    public boolean isEmpty() {
        Node<T> current = head.next.getReference();
        while (current != null) {
            if (!current.next.isMarked()) {
                return false;
            }
            current =  current.next.getReference();
        }
        return true;
    }

    @Override
    public Iterator<T> iterator() {
        return null;
    }
}
