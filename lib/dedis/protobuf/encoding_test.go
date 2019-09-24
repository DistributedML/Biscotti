package protobuf

import (
	"encoding"
	"testing"

	"github.com/stretchr/testify/assert"
)

type Number interface {
	encoding.BinaryMarshaler
	encoding.BinaryUnmarshaler

	Value() int
}

type Int struct {
	N int
}

type Wrapper struct {
	N Number
}

func NewNumber(n int) Number {
	return &Int{n}
}

func (i *Int) Value() int {
	return i.N
}

func (i *Int) MarshalBinary() ([]byte, error) {
	return []byte{byte(i.N)}, nil
}

func (i *Int) UnmarshalBinary(data []byte) error {
	i.N = int(data[0])
	return nil
}

// Check at compile time that we satisfy the interfaces.
var _ encoding.BinaryMarshaler = (*Int)(nil)
var _ encoding.BinaryUnmarshaler = (*Int)(nil)

// Validate that support for self-encoding via the Encoding
// interface works as expected
func TestBinaryMarshaler(t *testing.T) {
	wrapper := Wrapper{NewNumber(99)}
	buf, err := Encode(&wrapper)
	assert.Nil(t, err)

	wrapper2 := Wrapper{NewNumber(0)}
	err = Decode(buf, &wrapper2)

	assert.Nil(t, err)
	assert.Equal(t, 99, wrapper2.N.Value())
}

type NumberNoMarshal interface {
	Value() int
}

func NewNumberNoMarshal(n int) NumberNoMarshal {
	return &IntNoMarshal{n}
}

type IntNoMarshal struct {
	N int
}

func (i *IntNoMarshal) Value() int {
	return i.N
}

type WrapperNoMarshal struct {
	N NumberNoMarshal
}

func TestNoBinaryMarshaler(t *testing.T) {
	wrapper := WrapperNoMarshal{NewNumberNoMarshal(99)}
	buf, err := Encode(&wrapper)
	assert.Nil(t, err)

	wrapper2 := WrapperNoMarshal{NewNumberNoMarshal(0)}
	err = Decode(buf, &wrapper2)

	assert.Nil(t, err)
	assert.Equal(t, 99, wrapper2.N.Value())
}

type WrongSliceInt struct {
	Ints [][]int
}
type WrongSliceUint struct {
	UInts [][]uint16
}

func TestNo2dSlice(t *testing.T) {
	w := &WrongSliceInt{}
	w.Ints = [][]int{[]int{1, 2, 3}, []int{4, 5, 6}}
	_, err := Encode(w)
	assert.NotNil(t, err)

	w2 := &WrongSliceUint{}
	w2.UInts = [][]uint16{[]uint16{1, 2, 3}, []uint16{4, 5, 6}}
	_, err = Encode(w2)
	assert.NotNil(t, err)
}

type T struct {
	Buf1, Buf2 []byte
}

func TestByteOverwrite(t *testing.T) {
	t0 := T{
		Buf1: []byte("abc"),
		Buf2: []byte("def"),
	}
	buf, err := Encode(&t0)
	assert.Nil(t, err)

	var t1 T
	err = Decode(buf, &t1)
	assert.Nil(t, err)

	assert.Equal(t, []byte("abc"), t1.Buf1)
	assert.Equal(t, []byte("def"), t1.Buf2)

	// now we trigger the bug that used to exist, by writing off the end of
	// Buf1, over where the size was (the g and h) and onto the top of Buf2.
	b1 := append(t1.Buf1, 'g', 'h', 'i')
	assert.Equal(t, []byte("abcghi"), b1)
	// Buf2 must be unchanged, even though Buf1 was written to. When the bug
	// was present, Buf2 turns into "ief".
	assert.Equal(t, []byte("def"), t1.Buf2)

	// With the fix in place, the capacities must match the lengths.
	assert.Equal(t, len(t1.Buf1), cap(t1.Buf1))
	assert.Equal(t, len(t1.Buf2), cap(t1.Buf2))
}
