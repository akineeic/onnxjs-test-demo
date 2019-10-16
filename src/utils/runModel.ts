import { InferenceSession, Tensor } from 'onnxjs';

export function createSession(session: InferenceSession | undefined, hint: string): boolean {
    if (session) {
        return false;
    } 
    session = new InferenceSession({backendHint: hint});
    return true;
}

export async function warmupModel(model: InferenceSession, dims: number[]) {
    // OK. we generate a random input and call Session.run() as a warmup query
    const size = dims.reduce((a, b) => a * b);
    const warmupTensor = new Tensor(new Float32Array(size), 'float32', dims);

    for (let i = 0; i < size; i++) {
        warmupTensor.data[i] = Math.random() * 2.0 - 1.0; // random value [-1.0, 1.0)
    }
    try {
        await model.run([warmupTensor]);
    } catch (e) {
        console.error(e);
    }

}

export async function runModel(model: InferenceSession, preprocessedData: Tensor): Promise<[Tensor, number]> {
    try {
        const sumTime: number[] = [];
        const iterations = 200;
        for (let i = 0; i < iterations+1; i++){
            const start = performance.now();
            await model.run([preprocessedData]);
            const end = performance.now();
            const inferenceTime = end - start;
            sumTime.push(inferenceTime);
            console.log(`Iteration: ${i+1} / ${iterations+1}, InferenceTime: ${inferenceTime.toFixed(2)}`);
        }

        const mean = calMean(sumTime);
        const std = calStd(sumTime);
        console.log(`InferenceTime: ${mean.toFixed(2)} ± ${std.toFixed(2)} [ms]`);

        const start = new Date();
        const outputData = await model.run([preprocessedData]);
        const end = new Date();
        const inferenceTime = (end.getTime() - start.getTime());
        const output = outputData.values().next().value;

        return [output, inferenceTime];
    } catch (e) {
        console.error(e);
        throw new Error();
    }
    
}

    function calMean(results: number[]): number{
            // remove first run, which is regarded as "warming up" execution
            results.shift();
            const d = results.reduce((d, v) => {
                d.sum += v;
                d.sum2 += v * v;
                return d;
            }, {
                sum: 0,
                sum2: 0
            });
            const mean = d.sum / results.length;
            //let std = Math.sqrt((d.sum2 - results.length * mean * mean) / (results.length - 1));
            return mean;
    }

    function calStd(results: number[]): number{
        // remove first run, which is regarded as "warming up" execution
        results.shift();
        const d = results.reduce((d, v) => {
            d.sum += v;
            d.sum2 += v * v;
            return d;
        }, {
            sum: 0,
            sum2: 0
        });
        const mean = d.sum / results.length;
        const std = Math.sqrt((d.sum2 - results.length * mean * mean) / (results.length - 1));
        return std;
}
