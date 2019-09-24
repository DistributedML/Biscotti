/*
Package schnorr implements the vanilla Schnorr signature scheme.
See https://en.wikipedia.org/wiki/Schnorr_signature.

The only difference regarding the vanilla reference is the computation of
the response. This implementation adds the random component with the
challenge times private key while the Wikipedia article substracts them.

The resulting signature is compatible with EdDSA verification algorithm
when using the edwards25519 group, and by extension the CoSi verification algorithm.
*/
package schnorr

import (
	"bytes"
	"crypto/sha512"
	"errors"
	"fmt"

	"github.com/dedis/kyber"
)

// Suite represents the set of functionalities needed by the package schnorr.
type Suite interface {
	kyber.Group
	kyber.Random
}

// Sign creates a Sign signature from a msg and a private key. This
// signature can be verified with VerifySchnorr. It's also a valid EdDSA
// signature when using the edwards25519 Group.
func Sign(s Suite, private kyber.Scalar, msg []byte) ([]byte, error) {
	var g kyber.Group = s
	// create random secret k and public point commitment R
	k := g.Scalar().Pick(s.RandomStream())
	R := g.Point().Mul(k, nil)

	// create hash(public || R || message)
	public := g.Point().Mul(private, nil)
	h, err := hash(g, public, R, msg)
	if err != nil {
		return nil, err
	}

	// compute response s = k + x*h
	xh := g.Scalar().Mul(private, h)
	S := g.Scalar().Add(k, xh)

	// return R || s
	var b bytes.Buffer
	if _, err := R.MarshalTo(&b); err != nil {
		return nil, err
	}
	if _, err := S.MarshalTo(&b); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// Verify verifies a given Schnorr signature. It returns nil iff the
// given signature is valid.
func Verify(g kyber.Group, public kyber.Point, msg, sig []byte) error {
	R := g.Point()
	s := g.Scalar()
	pointSize := R.MarshalSize()
	scalarSize := s.MarshalSize()
	sigSize := scalarSize + pointSize
	if len(sig) != sigSize {
		return fmt.Errorf("schnorr: signature of invalid length %d instead of %d", len(sig), sigSize)
	}
	if err := R.UnmarshalBinary(sig[:pointSize]); err != nil {
		return err
	}
	if err := s.UnmarshalBinary(sig[pointSize:]); err != nil {
		return err
	}
	// recompute hash(public || R || msg)
	h, err := hash(g, public, R, msg)
	if err != nil {
		return err
	}

	// compute S = g^s
	S := g.Point().Mul(s, nil)
	// compute RAh = R + A^h
	Ah := g.Point().Mul(h, public)
	RAs := g.Point().Add(R, Ah)

	if !S.Equal(RAs) {
		return errors.New("schnorr: invalid signature")
	}

	return nil
}

func hash(g kyber.Group, public, r kyber.Point, msg []byte) (kyber.Scalar, error) {
	h := sha512.New()
	if _, err := r.MarshalTo(h); err != nil {
		return nil, err
	}
	if _, err := public.MarshalTo(h); err != nil {
		return nil, err
	}
	if _, err := h.Write(msg); err != nil {
		return nil, err
	}
	return g.Scalar().SetBytes(h.Sum(nil)), nil
}
