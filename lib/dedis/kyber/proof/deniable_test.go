package proof

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/dedis/kyber"
	"github.com/dedis/kyber/group/edwards25519"
	"github.com/dedis/kyber/util/random"
)

var testSuite = edwards25519.NewBlakeSHA256Ed25519()

type node struct {
	i    int
	done bool

	x kyber.Scalar
	X kyber.Point

	proto  Protocol
	outbox chan []byte
	inbox  chan [][]byte

	log *bytes.Buffer
}

func (n *node) Step(msg []byte) ([][]byte, error) {

	n.outbox <- msg
	msgs := <-n.inbox
	return msgs, nil
}

func (n *node) Random() kyber.XOF {
	return testSuite.XOF([]byte("test seed"))
}

func runNode(n *node) {
	errs := (func(Context) []error)(n.proto)(n)

	fmt.Fprintf(n.log, "node %d finished\n", n.i)
	for i := range errs {
		if errs[i] == nil {
			fmt.Fprintf(n.log, "- %d: SUCCESS\n", i)
		} else {
			fmt.Fprintf(n.log, "- %d: %s\n", i, errs[i])
		}
	}

	n.done = true
	n.outbox <- nil
}

func TestDeniable(t *testing.T) {
	nnodes := 5

	suite := testSuite
	rand := random.New()
	B := suite.Point().Base()

	// Make some keypairs
	nodes := make([]*node, nnodes)
	for i := 0; i < nnodes; i++ {
		n := &node{}
		nodes[i] = n
		n.i = i
		n.x = suite.Scalar().Pick(rand)
		n.X = suite.Point().Mul(n.x, nil)
		n.log = &bytes.Buffer{}
	}

	// Make some provers and verifiers
	for i := 0; i < nnodes; i++ {
		n := nodes[i]
		pred := Rep("X", "x", "B")
		sval := map[string]kyber.Scalar{"x": n.x}
		pval := map[string]kyber.Point{"B": B, "X": n.X}
		prover := pred.Prover(suite, sval, pval, nil)

		vi := (i + 2) % nnodes // which node's proof to verify
		vrfs := make([]Verifier, nnodes)
		vpred := Rep("X", "x", "B")
		vpval := map[string]kyber.Point{"B": B, "X": nodes[vi].X}
		vrfs[vi] = vpred.Verifier(suite, vpval)

		n.proto = DeniableProver(suite, i, prover, vrfs)
		n.outbox = make(chan []byte)
		n.inbox = make(chan [][]byte)

		go runNode(n)
	}

	for {
		// Collect messages from all still-active nodes
		msgs := make([][]byte, nnodes)
		done := true
		for i := range nodes {
			n := nodes[i]
			if n == nil {
				continue
			}
			done = false
			msgs[i] = <-n.outbox

			if n.done {
				t.Log(string(n.log.Bytes()))
				nodes[i] = nil
			}
		}
		if done {
			break
		}

		// Distribute all messages to all still-active nodes
		for i := range nodes {
			if nodes[i] != nil {
				nodes[i].inbox <- msgs
			}
		}
	}
}
