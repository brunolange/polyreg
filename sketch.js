(function (tf, window, document) {
let resetButton
let dragging = false

let xs = []
let ys = []

let degree = 4
let coeff = []
let X, Y
const X_size = 100
const X_max = 3
const X_min = -3
const Y_max = 3
const Y_min = -3

const learningRate = 0.1
const optimizer = tf.train.adam(learningRate)

let loss = (guess, actual) => {
    return guess.sub(actual).square().mean()
}

let predict = (xs) => {
    // X = Array(0).fill(xs.size) // WebGL buffer error!!!
    let partial = tf.zeros([1, xs.size])
    if (xs.size) {
        for (let i=0; i<=degree; i++) {
            partial = partial.add(
                xs.pow(tf.scalar(i)).mul(coeff[i])
            )
        }
    }
    return partial
}

function setup() {
    createCanvas(700, 700)

    // initialize polynomial coefficients to random values between -1 and 1
    for (let i=0; i<=degree; i++) {
        coeff.push(tf.variable(tf.scalar(random(-0.5, 0.5))))
    }

    // initialize domain
    X = Array(X_size).fill(X_min)
    for (let i=1; i<X_size; i++) {
        X[i] = X[i-1] + (X_max - X_min)/(X_size-1)
    }
    print(X)

    // UI
    resetButton = createButton('reset');
    resetButton.position(10, height+20);
    resetButton.mousePressed(reset);
}

function reset() {
    xs = []
    ys = []
}

function mousePressed() {
    dragging = true
}

function mouseReleased() {
    dragging = false
}

function drawGrid() {
    stroke(220)
    strokeWeight(4)
    line(width/2, 0, width/2, height)
    line(0, height/2, width, height/2)
    strokeWeight(1)
    for (let j=0; j<X_max; j++) {
        let dx = (j+1)*(width/2)/X_max
        line(width/2 + dx, 0, width/2 + dx, height)
        line(width/2 - dx, 0, width/2 - dx, height)
    }
    for (let i=0; i<X_max; i++) {
        let dy = (i+1)*(height/2)/X_max
        line(0, height/2 + dy, width, height/2 + dy)
        line(0, height/2 - dy, width, height/2 - dy)
    }
}

function draw() {
    if (dragging) {
        const x = map(mouseX, 0, width, X_min, X_max)
        const y = map(mouseY, 0, height, Y_max, Y_min)
        if (x < X_min || x > X_max) return
        if (y < Y_min || y > Y_max) return

        xs.push(x)
        ys.push(y)
    } else {
        if (xs.length > degree) {
            // minimize loss with respect to coeff
            t_xs = tf.tensor1d(xs)
            t_ys = tf.tensor1d(ys)
            optimizer.minimize(() => loss(predict(t_xs), t_ys))
            t_xs.dispose()
            t_ys.dispose()
        }
    }
    background(0)

    drawGrid()
    stroke(255, 0, 0);
    strokeWeight(6);
    for (let i = 0; i < xs.length; i++) {
        let px = map(xs[i], X_min, X_max, 0, width);
        let py = map(ys[i], Y_min, Y_max, height, 0);
        point(px, py);
    }

    const T_Y = tf.tidy(() => predict(tf.tensor1d(X)))
    Y = T_Y.dataSync()
    T_Y.dispose()

    beginShape()
    noFill()
    stroke(255,255,0)
    strokeWeight(4)
    for (let i=0; i<X.length; i++) {
        let x = map(X[i], X_min, X_max, 0, width)
        let y = map(Y[i], Y_min, Y_max, height, 0)
        vertex(x, y)
    }
    endShape()
}

// export p5.js functions
window.setup = setup
window.draw = draw
window.mousePressed = mousePressed
window.mouseReleased = mouseReleased
// export coefficients
window.coeff = coeff

})(tf, window, document, undefined)
