package protobuf

import (
	"testing"
	"fmt"
	"github.com/stretchr/testify/assert"
)

type ValueInt interface{
	Print()string
}

type A struct{
	Value int
}

func (a *A)MarshalBinary()([]byte, error){
	return []byte(fmt.Sprintf("%d", a.Value)), nil
}

func (a *A)Print()string{
	return ""
}

type B struct{
	AValue A
	AInt int
}

func TestMarshal(t *testing.T) {
	var a A = A{0}
	var b B = B{a, 1}

	assert.True(t, isBinaryMarshaler(a))
	assert.False(t, isBinaryMarshaler(b))

	bufA, _ := Encode(&a)
	bufB, _ := Encode(&b)

	t.Log(bufA)
	t.Log(bufB)

	testA := A{}
	testB := B{}

	assert.True(t, isBinaryMarshaler(testA))
	assert.False(t, isBinaryMarshaler(testB))

	Decode(bufA, &testA)
	Decode(bufB, &testB)

	assert.Equal(t, testA, a)
	assert.Equal(t, testB, b)
}
