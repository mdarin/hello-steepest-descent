/*
 * Иллюстрация метода наискорейшего спуска
 * 
 * go run hello_steepest_descent.go
 */
package main

import(
	"fmt"
	"math"
)

/*
  __
  \/F - оператор набла nabla F
  или другое обозначение grad F


  Градиентный спуск
  Алгоритм
	1.Задают начальное приближение и точность расчёта vec x[0],epsilon 
	2.Рассчитывают vec x[k+1]=vec x[k] - lambda[k] * nabla F(vec x[k]), 
	  где lambda[k] = argmin_lambda F(vec x[k] - lambda * nabla F(vec x[k]))
	3.Проверяют условие остановки:
	  Если |vec x[k+1] - vec x[k]| > epsilon 
	    или |F(vec x[k+1]) - F(vec x[k])| > epsilon  
		или |nabla F(vec x[k+1])| > epsilon (выбирают одно из условий), 
		то k = k + 1 и переход к пункту 2.
	  Иначе vec x = vec x[k+1] и останов.

	[https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B4%D0%B8%D0%B5%D0%BD%D1%82%D0%BD%D1%8B%D0%B9_%D1%81%D0%BF%D1%83%D1%81%D0%BA]

  Алгоритм метода наискорейшего спуска
	Шаг 1. Задать x[0], epsilon1 > 0,epsilon2 > 0, предельное число итераций М. 
			Найти градиент функции в произвольной точке.
			Найти частные производные функции f(x)
			urad f(x) = [ df(x)/x1,...,df(x)/dxn ]T(транспонированный вектор)
	Шаг 2. Положить k = 0.
	Шаг 3. Вычислить grad f(x[k]).
	Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
		а) если критерий выполнен, то x[0] = x[k], останов;
		б) если критерий не выполнен, то перейти к шагу 5.
	Шаг 5. Проверить выполнение неравенства k ≥ M:
		а) если неравенство выполнено, то x[0] = x[k], останов;
		б) если нет, то перейти к шагу 6.
	Шаг 6. Вычислить величину шага lambda[k=0], из условия
		F(lambda[k]) = f(x[k] - lambda[k] * grad f(x[k])) -> min_lamda[k].
	Шаг 7. Вычислить x[k+1] = x[k] - lambda[k] * grad f(x[k]).
	Шаг 8. Проверить выполнение условий ||x[k+1] - x[k]|| < epsilon2, ||f(x[k+1]) - f(x[k])|| < epsilon2:
		а) если оба условия выполнены при текущем значении k и k = k - 1, то расчет окончен, x[0] = x[k+1];
		б) если хотя бы одно из условий не выполнено, то положить k = k + 1 и перейти к шагу 3.

	[https://studfiles.net/preview/1874323/page:2/]
*/

//иллюстрация реализации интерфейса для метода наискорейшего спуска и для методов одномерной оптимизации
type Mydatainterface interface {
	// функция g(x) оценки минимума в методах одномерной оптимизации 
	// в методе наискорейшего спуска(вообще в многомерной оптимизации) это функция g(x[k]) 
	// применяемая для нахождения шага поиска lambda[k] как минимума функции g(x[k])
	// на отрезке a,b 
	// lambda здесь это вычисленная точка по методу поиска минимума, имя надо бы выбрать другое... 
	GetG() func(data OMInterface, lambda float64) float64
	// фукнция миниму которой мы ищев в методах многомерной оптимизации
	Function() float64
	//GetFunction() (func(data OMInterface) float64)
	// вектор аргументов функции
	GetX() []float64
	SetX(args []float64)
	// значение фукнции
	SetY(float64)
	GetY() float64
	// градиет функции - вектор частных производных
	Grad() []float64
	// отрезок для поиска минимума методом одной оптимизации
	//a,b float64
	GetInterval() (float64, float64)
	// максимальное количесво приближений 
	//n int - для одномерной оптимизации
	//m int - многомерной оптимизации
	GetMaxIter1DOpt() int
	GetMaxIterMDOpt() int
	// требуемая точность
	//epsilon1 float64 для метода одномерной оптимизации
	//epsilon2 float64 для метода многомерной оптимизации
	GetEpsilon1() float64
	GetEpsilon2() float64
	// Аргумент минимизации (для наискорейшего спуска argmin по lambda)
	// phi(x) метод одномерной оптимизаци
	GetF() (func(data OMInterface) float64)

}
type Mydata struct {
	args []float64
	y float64
}
func New(n int) *Mydata {
	md := new(Mydata)
	md.args = make([]float64, n)
	return md
}

func (md *Mydata) Function() float64 {
	return md.args[0]*md.args[0] + 2.0*md.args[0]*md.args[1] + 3.0*md.args[1]*md.args[1] - 2.0*md.args[0]-3.0*md.args[1]
}

/*func (md *Mydata) GetFunction() (func (data OMInterface) float64) {
	return func (data OMInterface) float64 {
		args := data.GetX()
		return args[0]*args[0] + 2.0*args[0]*args[1] + 3.0*args[1]*args[1] - 2.0*args[0]-3.0*args[1]
	}
}*/

func (md *Mydata) GetX() []float64 {
	return md.args
}

func (md *Mydata) SetX(args []float64) {
	copy(md.args, args)
}

func (md *Mydata) SetY(y float64) {
	md.y = y
}

func (md *Mydata) GetY() float64 {
	return md.y
}

func (md *Mydata) GetEpsilon() float64 { return 0.01 }

func (md *Mydata) Grad() []float64 {
	args := md.GetX()
	//function := md.GetFunction()
	left := New(len(args))
	left.SetX(args)
	right := New(len(args))
	right.SetX(args)
	delta := 0.05
	gradient := make([]float64, len(args))
	for i := 0; i < len(gradient); i++ {
		left.args[i] += delta
		right.args[i] -= delta
		gradient[i] = ( left.Function() - right.Function() ) / (2.0 * delta)
	}
	return gradient

}

// задать фукнцию оценки g(x) для метода одномерной оптимизации
func (md *Mydata) GetG() (func (data OMInterface, lambda float64) float64) {
	g := func (data OMInterface, lambda float64) float64 {
		args := data.GetX()
		//function := data.GetFunction()
		temp := New(len(args))
		gradient := data.Grad()
		for i,df_dxi := range gradient {
			temp.args[i] = args[i] - lambda * df_dxi
		}
		return temp.Function()
	}
	return g
}

func (md *Mydata) GetInterval() (float64,float64) {
	a, b := -10000.0, 100000.0
	return a,b
}
// максимальное количесво приближений 
//n int - для одномерной оптимизации
//m int - многомерной оптимизации
func (md *Mydata) GetMaxIter1DOpt() (n int) {
	n = 100
	return
}
func (md *Mydata) GetMaxIterMDOpt() (m int) {
	m = 1000
	return
}
// требуемая точность
//epsilon1 float64 для метода одномерной оптимизации
//epsilon2 float64 для метода многомерной оптимизации
func (md *Mydata) GetEpsilon1() float64 { return 0.01 }
func (md *Mydata) GetEpsilon2() float64 { return 0.01 }
// Аргумент минимизации (для наискорейшего спуска argmin по lambda)
// phi(x) метод одномерной оптимизаци
func (md *Mydata) GetF() (func(data OMInterface) float64) {
	return dichotomia3
}



//
// main driver
//
func main() {
	fmt.Println("Hello steepest descent")
	// совсем простой пример
	//TODO: эти значения можно менять и наблюдать результат
	var x float64 = 34.0
	var y float64 = 12.0
	var epsilon = 0.01
  SteepestDescent(x, y, epsilon)

	// уже посложней с функциями-аргументами
	fn := func (args []float64) float64 {
		//return x[0]*x[0] + 23.0*x[0]
		return args[0]*args[0] + 2.0*args[0]*args[1] + 3.0*args[1]*args[1] - 2.0*args[0]-3.0*args[1]
	}
	SteepestDescent2(fn, []float64{x,y}, epsilon)
	fmt.Println()

	// здесь уже использованы интерфейсы и фунции высшего порядка
	// не понятно, сложно или всё же универсально вышло?
	mydata := New(2)
	mydata.SetX([]float64{x,y})
	SteepestDescent3(mydata)
	fmt.Printf("## x: ")
	for _,x := range mydata.GetX() {
		fmt.Printf(" %.3f", x)
	}
	fmt.Println()
	fmt.Printf("## y: %.3f\n", mydata.GetY())
}

// Собственно здесь записывается наша функция
func f(x,y float64) float64 {
	//TODO: здеась можно задать исследуемею функцию или поиграться с этой
	//NOTE: если поменяете функцию не забудьте про частные производные
    return x*x + 2.0*x*y + 3.0*y*y - 2.0*x -3.0*y
}

// Это первая производная по dx
// частная производная по х
func f_dx(x, y float64) float64 {
    return 2.0*x + 2.0*y - 2.0
}

// Это первая производная по dy
// частная производная по y
func f_dy(x, y float64) float64 {
    return 2.0*x + 6.0*y - 3.0
}


// Это функция g в методе наискорейшего (градиентного) спуска
// операция grad или nabla
func g(x, y, lambda float64) float64 {

	//var model_f_dx float64 = f_dx(x, y)
	//var model_f_dy float64 = f_dy(x, y)
	// срединный разностный метод вычисления первой производной
	// numerical differentiation
	// NOTE: delta ought to be small enough but you should remember 
	//       that too small value will drive to reducing accuracy
	//var delta float64 = 0.5
	// nabla f(x,y)
	//var approx_f_dx float64 = ( f(x + delta / 2.0, y) - f(x - delta / 2.0, y) ) / delta 
	// nabla f(x,y)
	//var approx_f_dy float64 = ( f(x, y + delta / 2.0) - f(x, y - delta / 2.0) ) / delta 	
	//var approx_f_dx float64 = ( f(x + delta, y) - f(x - delta, y) ) / (2.0 * delta) 
	//var approx_f_dy float64 = ( f(x, y + delta) - f(x, y - delta) ) / (2.0 * delta)

	//fmt.Printf("  model x: %.5f  approx1 x: %.5f\n", model_f_dx, approx_f_dx)
	//fmt.Printf("  model y: %.5f  approx1 y: %.5f\n", model_f_dy, approx_f_dy)
	//fmt.Println()

	//	Шаг 1. (продолжение)
	//		Найти градиент функции в произвольной точке. Определить частные производные функции f(x):
	//		grad f(x) = [ df(x)/x1,...,df(x)/dxn ]T(транспонированный вектор)
    return f(x - lambda * f_dx(x, y), y - lambda * f_dy(x, y))
	//return f(x - lambda * approx_f_dx, y - lambda * approx_f_dy)
}


func g2(function func (x float64) float64, x, lambda, delta float64) float64 {
	return function(x - lambda * grad(function, x, delta))
}
// срединный разностный метод вычисления первой производной
// numerical differentiation
// median difference method
//                      __
// grad F or nabla F or \/F
func grad(function func (x float64) float64, x, delta float64) float64 {
	// NOTE: delta ought to be small enough but you should remember 
	//       that too small value will drive to reducing accuracy
	return ( function(x + delta / 2.0) - function(x - delta / 2.0) ) / delta
	//return ( function(x + delta) - function(x - delta) ) / (2.0 * delta)
}
// Найти градиент функции в произвольной точке.
// Найти частные производные функции f(x)
// grad f(x) = [ df(x)/x1,...,df(x)/dxn ]T(транспонированный вектор)
//                      __
// grad F or nabla F or \/F
func grad2(function func (args []float64) float64, args []float64, delta float64) []float64 {
	// NOTE: delta ought to be small enough but you should remember 
	//       that too small value will drive to reducing accuracy
	//
	// df/dxi = f(x1,x2,...,xi+/\xi,...xn) - f(x1,x2,...xi-/\xi,...xn) / (2 * /\xi) 
	//
	gradient := make([]float64, len(args))
	for i := 0; i < len(gradient); i++ {
		left := make([]float64, len(args))
		copy(left, args)
		right := make([]float64, len(args))
		copy(right, args)
		left[i] += delta
		right[i] -= delta
		//fmt.Println(" left:", left)
		//fmt.Println(" right:", right)
		//fmt.Println(" args after:", args)
		gradient[i] = ( function(left) - function(right) ) / (2.0 * delta)
	}
	return gradient
}


// двумерная норма
// условие остановки
func norma(x, y float64) float64 {
    return math.Sqrt(x*x + y*y)
}

func norma2(x float64) float64 {
    return math.Sqrt(x*x)
}


// Метод половинного деления для нахождения минимума в градиентном спуске
// dichotomia — разделение на две части
func dichotomia(a0, b0, epsilon, x, y float64) float64 {
    // Номер шага
		var k int
    // Отклонени от середины отрезка влево, вправо
    var left float64
		var right float64
    // Величина на которую мы отклонимся от середины отрезка
    var deviation float64 = 0.5 * epsilon
    // Точка минимума
    var x_min float64
    // Отрезок локализации минимума
    var ak float64 = a0
		var bk float64 = b0

		//	Шаг 3. Вычислить grad f(x[k]).
		//	Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
		//		а) если критерий выполнен, то x[0] = x[k], останов;
		//		б) если критерий не выполнен, то перейти к шагу 5.
    // Пока длина отрезка больше заданной точности
    for k = 1; (bk - ak) >= epsilon; k++ {
        // Берем середину (ну почти середину - +\- некоторое дельта 
				// в частности у нас deviation = 0.5 * epsilon)
        left = (ak + bk - deviation) / 2.0
        right = (ak + bk + deviation) / 2.0
				// Шаг 3. Вычислить grad f(x[k])
        // Проверяем в какую часть попадает точка минимума 
				// слева от разбиения или справа и выбираем соответствующую точку
        if g(x, y, left) <= g(x, y, right) {
        // Теперь правая граница отрезка локализации равна right
            bk = right;
        } else {
        // Теперь левая граница отрезка локализации равна left
            ak = left;
        }
    }

	// minimum point
    x_min = (ak + bk) / 2.0

    return x_min;
}



// Метод половинного деления для нахождения минимума
// dichotomia — разделение на две части
// epsiolon критерий требуемой точности
func dichotomia2(g func(args []float64, lambda float64) float64, args []float64, a0, b0, epsilon float64/*,n int //max iterations*/) float64 {
	// Номер шага
	var k int
	// Отклонени от середины отрезка влево, вправо
	var left float64
	var right float64
	// Величина на которую мы отклонимся от середины отрезка
	var deviation float64 = 0.5 * epsilon
	// Точка минимума
	var x_min float64
	// Отрезок локализации минимума
	var ak float64 = a0
	var bk float64 = b0

	//	Шаг 3. Вычислить grad f(x[k]).
	//	Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
	//		а) если критерий выполнен, то x[0] = x[k], останов;
	//		б) если критерий не выполнен, то перейти к шагу 5.
	// Пока длина отрезка больше заданной точности
	for k = 1; (bk - ak) >= epsilon; k++ {
		// Берем середину (ну почти середину - +\- некоторое дельта 
		// в частности у нас deviation = 0.5 * epsilon)
		left = (ak + bk - deviation) / 2.0
		right = (ak + bk + deviation) / 2.0
		//fmt.Printf(" [%d] real left: %.3f  right: %.3f epsilon: %f\n", k, left, right, (right - left))
		//fmt.Println()
		// Шаг 3. Вычислить grad f(x[k])
		// Проверяем в какую часть попадает точка минимума 
		// слева от разбиения или справа и выбираем соответствующую точку
		if g(args, left) <= g(args, right) {
		// Теперь правая граница отрезка локализации равна right
				bk = right;
		} else {
		// Теперь левая граница отрезка локализации равна left
				ak = left;
		}
	}

	// точка как серединка полученного отрезочка a b
	// minimum point
  x_min = (ak + bk) / 2.0;
	//fmt.Printf("  x_min: %.3f\n", x_min)
    return x_min;
}

// Метод половинного деления для нахождения минимума
// dichotomia — разделение на две части
// epsiolon критерий требуемой точности
func dichotomia3(data OMInterface) float64 {
	// Номер шага
	var k int
	// Отклонени от середины отрезка влево, вправо
	var left float64
	var right float64
	// Величина на которую мы отклонимся от середины отрезка
	epsilon := data.GetEpsilon1()
	var deviation float64 = 0.5 * epsilon
	// Точка минимума
	var x_min float64
	// Отрезок локализации минимума
	ak,bk := data.GetInterval()
	// функция g(x) оценки минимума в методах одномерной оптимизации на отрезке a,b 
	g := data.GetG()

	//	Шаг 3. Вычислить grad f(x[k]).
	//	Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
	//		а) если критерий выполнен, то x[0] = x[k], останов;
	//		б) если критерий не выполнен, то перейти к шагу 5.
	// Пока длина отрезка больше заданной точности
	for k = 1; (bk - ak) >= epsilon; k++ {
		// Берем середину (ну почти середину - +\- некоторое дельта 
		// в частности у нас deviation = 0.5 * epsilon)
		left = (ak + bk - deviation) / 2.0
		right = (ak + bk + deviation) / 2.0
		//fmt.Printf(" [%d] real left: %.3f  right: %.3f epsilon: %f\n", k, left, right, (right - left))
		//fmt.Println()
		// Шаг 3. Вычислить grad f(x[k])
		// Проверяем в какую часть попадает точка минимума 
		// слева от разбиения или справа и выбираем соответствующую точку
		// и переносим границу
		if g(data, left) <= g(data, right) {
		// Теперь правая граница отрезка локализации равна right
			bk = right;
		} else {
		// Теперь левая граница отрезка локализации равна left
			ak = left;
		}
	}

	// точка как серединка полученного отрезочка a b
	// minimum point
  x_min = (ak + bk) / 2.0;
	//fmt.Printf("  x_min: %.3f\n", x_min)
  return x_min;
}

// Метод деления в пропорции "Золотого сечения" для нахождения минимума
// epsiolon критерий требуемой точности
func goldensection(g func(args []float64, lambda float64) float64, args []float64, a0, b0, epsilon float64/*,n int //max iterations*/) float64 {
  // Номер шага
	var k int
	// Отклонени от точки деления отрезка влево, вправо
	var left float64
	var right float64
	// Точка минимума
	var x_min float64
	// Отрезок локализации минимума
	var ak float64 = a0
	var bk float64 = b0
	// Пропорция золотого сечения
    // L / m = m / n
	// L = 1 - это целый отрезок
	// m = (-1+sqrt(5))/2 = 0,618*L - это бОльшая часть
	// n = L - m = 0,382 * L - это меньшая часть
	m := (-1.0 + math.Sqrt(5.0)) / 2.0
	n := 1.0 - m
	//fmt.Printf("K = %.3f 1-K = %.3f\n", m, n)

	//	Шаг 3. Вычислить grad f(x[k]).
	//	Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
	//		а) если критерий выполнен, то x[0] = x[k], останов;
	//		б) если критерий не выполнен, то перейти к шагу 5.
    // Пока длина отрезка больше заданной точности
	niterations := 100
  for k = 1; (bk - ak) >= epsilon && k < niterations; k++ {
		// Деллим отрезок в пропрорции золотого сечения
		left = ak + (bk - ak) * n
		right = ak + (bk - ak) * m
		//fmt.Printf(" [%d] model left: %.3f  right: %.3f epsilon: %f\n", k, mleft, mright, (mright - mleft))
		//fmt.Printf(" [%d] real left: %.3f  right: %.3f epsilon: %f\n", k, left, right, (right - left))
		//fmt.Println()

		// Шаг 3. Вычислить grad f(x[k])
        // Проверяем в какую часть попадает точка минимума 
		// слева от разбиения или справа и выбираем соответствующую точку
		if g(args, left) <= g(args, right) {
		// Теперь правая граница отрезка локализации равна right
				bk = right
		} else {
		// Теперь левая граница отрезка локализации равна left
				ak = left
		}
  }

	// точка как серединка полученного отрезочка a b
	// minimum point
  x_min = (ak + bk) / 2.0
	//fmt.Printf("  x_min: %.3f\n", x_min)
  return x_min;
}



// n-мерная норма
// условие остановки
func norma3(cur, next []float64) float64 {
	var sumSquares float64

	for i := 0; i < len(cur); i++ {
		sumSquares += (next[i] - cur[i])*(next[i] - cur[i])
	}

	//fmt.Println("    norma:", sumSquares)
	//fmt.Println("    norma:", math.Sqrt(sumSquares))
	return math.Sqrt(sumSquares)
}





// Optimization methods interface
type OMInterface interface {
	// функция g(x) оценки минимума в методах одномерной оптимизации 
	// в методе наискорейшего спуска(вообще в многомерной оптимизации) это функция g(x[k]) 
	// применяемая для нахождения шага поиска lambda[k] как минимума функции g(x[k])
	// на отрезке a,b
	// lambda здесь это вычисленная точка по методу поиска минимума, имя надо бы выбрать другое... 
	GetG() func(data OMInterface, lambda float64) float64
	// фукнция миниму которой мы ищев в методах многомерной оптимизации
	Function() float64
	//GetFunction() (func(data OMInterface) float64)
	// вектор аргументов функции
	GetX() []float64
	SetX(args []float64)
	// значение фукнции
	SetY(float64)
	// градиет функции - вектор частных производных
	Grad() []float64
	// отрезок для поиска минимума методом одной оптимизации
	//a,b float64
	GetInterval() (float64,float64)
	// максимальное количесво приближений 
	//n int - для одномерной оптимизации
	//m int - многомерной оптимизации
	GetMaxIter1DOpt() int
	GetMaxIterMDOpt() int
	// требуемая точность
	//epsilon1 float64 для метода одномерной оптимизации
	//epsilon2 float64 для метода многомерной оптимизации
	GetEpsilon1() float64
	GetEpsilon2() float64
	// Аргумент минимизации (для наискорейшего спуска argmin по lambda)
	// phi(x) метод одномерной оптимизаци
	GetF() (func(data OMInterface) float64)
}




// Метод наискорейшего спуска(пример алгоритма)
// the method of steepest descent or stationary-phase method or saddle-point method
func SteepestDescent3(data OMInterface) {
	const MAXITERATIONS = 1000
	var k int
	var lambda float64

	// П.1. Задают начальное приближение и точность расчёта vec x[0], epsilon
	// Шаг 1. Задать x[0], epsilon1 > 0, epsilon2 > 0, предельное число итераций М. 
	//  Найти градиент функции в произвольной точке. Определить частные производные функции f(x):
	//  grad f(x) = [ df(x)/x1,...,df(x)/dxn ]T(транспонированный вектор)
	// Начальное приближение u[0]
	args := data.GetX()
	epsilon := data.GetEpsilon2()
	u_cur := make([]float64, len(args))
	// Новое прилижение u[k]
	u_next := make([]float64, len(u_cur))

	copy(u_cur, args)

	fmt.Println("## Исходная точка:")
	for _,x := range u_cur {
		fmt.Printf(" %.4f", x)
	}
	fmt.Println()

	fmt.Println()
	fmt.Println("## Приближения:")
	//Шаг 2. Положить k = 0
	stop := false
	for k = 0; k < MAXITERATIONS && !stop ; k++ {
		fmt.Printf("%d: ", k)
		// П.2. Рассчитывают vec x[k+1] = vec x[k] - lambda[k] * nabla F(vec x[k]),
		// где lambda[k]= argmin{lambda} F(vec x[k] - lambda * nabla F(vec x[k]))
		// Шаг 3. Вычислить grad f(x[k]).
		gradient := data.Grad()
		// Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
		//  а) если критерий выполнен, то x[0] = x[k], останов;
		//  б) если критерий не выполнен, то перейти к шагу 5.
		//
		// Шаг 5. Проверить выполнение неравенства k ≥ M:
		//  а) если неравенство выполнено, то x[0] = x[k], останов;
		//  б) если нет, то перейти к шагу 6
		//
		// Шаг 6. Вычислить величину шага lambda[k] из условия
		//  F(lambda[k]) = f(x[k] - lambda[k] * grad f(x[k])) -> min lambda[k]
		// Аргумент минимизации argmin по lambda(получить его реализацию)
		argmin := data.GetF()
		// Находим lambda[k] как минимум функции g(x[k]) на отрезке a = -10000, b = 100000
		lambda = argmin(data)
		//fmt.Println("   lambda:", lambda)
		// Вычисляем u[k] новое прилижение
		//  Шаг 7. Вычислить x[k+1] = x[k] - lambda[k] * grad f(x[k])
		for i,df_dxi := range gradient {
			u_next[i] = u_cur[i] - lambda * df_dxi
		}
		//DEBUG
		fmt.Println()
		data.SetX(u_next)
		fmt.Printf("  f(x) = %.4f\n", data.Function())
		for _,x := range u_next {
			fmt.Printf(" %.4f", x)
		}
		fmt.Println()

		// П.3. Проверяют условие остановки:
		// Если |vec x[k+1] - vec x[k]| > epsilon 
		//  или |F(vec x[k+1]) - F(vec x[k])| > epsilon  
		//  или |nabla F(vec x[k+1])| > epsilon (выбирают одно из условий), 
		// то k = k + 1 и переход к П.2.
		// Иначе vec x = vec x[k+1] и останов.
		// 	Шаг 8. Проверить выполнение условий ||x[k+1] - x[k]|| < epsilon2, ||f(x[k+1]) - f(x[k])|| < epsilon2:
		//		а) если оба условия выполнены при текущем значении k и k = k - 1, то расчет окончен, x[0] = x[k+1], останов;
		//		б) если хотя бы одно из условий не выполнено, то положить k = k + 1 и перейти к шагу 3
		if k > 1 {
			// Проверяем условие остановки
			if norma3(u_cur,u_next) < epsilon {
				// останов
				stop = true
			}
		}

		copy(u_cur, u_next)
	}

	// DEBUG
	fmt.Println()
	fmt.Println("##* Точка минимума при epsilon:", epsilon)
	data.SetX(u_cur)
	fmt.Println(" f(x):", data.Function())
	for _,x := range u_cur {
		fmt.Printf(" %.4f", x)
	}
	fmt.Println()

	// получить минимум фуркции
	data.SetX(u_cur)
	data.SetY(data.Function())
}




// Метод наискорейшего спуска(пример алгоритма)
// the method of steepest descent or stationary-phase method or saddle-point method
func SteepestDescent2(function func(args []float64) float64, args []float64, epsilon float64) float64 {
	const MAXITERATIONS = 1000
	var k int
	var lambda float64
	var delta float64 // шаг дифференцирования
	
	// П.1. Задают начальное приближение и точность расчёта vec x[0], epsilon
	// Шаг 1. Задать x[0], epsilon1 > 0, epsilon2 > 0, предельное число итераций М.
	//  Найти градиент функции в произвольной точке. Определить частные производные функции f(x):
	//  grad f(x) = [ df(x)/x1,...,df(x)/dxn ]T(транспонированный вектор)
	// Начальное приближение u[0]
	u_cur := make([]float64, len(args))
	// Новое прилижение u[k]
	u_next := make([]float64, len(u_cur))
	
	copy(u_cur, args)
	delta = 0.05 //TODO: ajust


	fmt.Println("## Исходная точка:")
	for _,x := range u_cur {
		fmt.Printf(" %.4f", x)
	}
	fmt.Println()
	
	fmt.Println()
	fmt.Println("## Приближения:")
	//Шаг 2. Положить k = 0
	stop := false
	for k = 0; k < MAXITERATIONS && !stop ; k++ {
		fmt.Printf("%d: ", k)
		// П.2. Рассчитывают vec x[k+1] = vec x[k] - lambda[k] * nabla F(vec x[k]), 
		// где lambda[k]= argmin_lambda F(vec x[k] - lambda * nabla F(vec x[k]))
		//  Шаг 3. Вычислить grad f(x[k]).
		gradient := grad2(function, u_cur, delta)
		
		//  Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
		//   а) если критерий выполнен, то x[0] = x[k], останов;
		//   б) если критерий не выполнен, то перейти к шагу 5.
		//
		//  Шаг 5. Проверить выполнение неравенства k ≥ M:
		//   а) если неравенство выполнено, то x[0] = x[k], останов;
		//   б) если нет, то перейти к шагу 6
		//
		//  Шаг 6. Вычислить величину шага lambda[k^0] из условия
		//   F(lambda[k]) = f(x[k] - lambda[k] * grad f(x[k])) -> min lambda[k]
		//
		//
		// Выберем метод дихотомии как F(lambda[k])
		//argmin := dichotomia2
		// Выберем метод золотого сечения
		argmin := goldensection
		//TODO: число приближений(итераций)
		//n := 10
		g := func (args []float64, lambda float64) float64 {
			// Это функция g(x[k]) в методе наискорейшего (градиентного) спуска
			// применяемая для нахождения шага lambda[k] как минимума функции g(x[k])
			// на отрезке(здес a=-10000, b=100000, методом дихотомии)
			temp := make([]float64, len(args))
			gradient := grad2(function, u_cur, delta)
			for i,dfx_dxi := range gradient {
				temp[i] = args[i] - lambda * dfx_dxi
			}

			return function(temp)
		}

		// Находим lambda[k] как минимум функции g(x[k]) на отрезке -10000,100000
		lambda = argmin(g, u_cur, -10000, 100000, epsilon)
		//fmt.Println("   lambda:", lambda)
	
		// Вычисляем u[k] новое прилижение
		//  Шаг 7. Вычислить x[k+1] = x[k] - lambda[k] * grad f(x[k])
		for i,dfx_dxi := range gradient {
			u_next[i] = u_cur[i] - lambda * dfx_dxi
		}
		fmt.Println()
		fmt.Printf("  f(x) = %.4f\n", function(u_next))
		for _,x := range u_next {
			fmt.Printf(" %.4f", x)
		}
		fmt.Println()

		// П.3. Проверяют условие остановки:
		// Если |vec x[k+1] - vec x[k]| > epsilon
		//  или |F(vec x[k+1]) - F(vec x[k])| > epsilon
		//  или |nabla F(vec x[k+1])| > epsilon (выбирают одно из условий),
		// то k = k + 1 и переход к П.2.
		// Иначе vec x = vec x[k+1] и останов.
		//  Шаг 8. Проверить выполнение условий ||x[k+1] - x[k]|| < epsilon2, ||f(x[k+1]) - f(x[k])|| < epsilon2:
		//   а) если оба условия выполнены при текущем значении k и k = k - 1, то расчет окончен, x[0] = x[k+1], останов;
		//   б) если хотя бы одно из условий не выполнено, то положить k = k + 1 и перейти к шагу 3
		if k > 1 {
			// Проверяем условие остановки
			if norma3(u_cur,u_next) < epsilon {
				// останов
				stop = true
			}
		}
		
		copy(u_cur, u_next)
	}
	
	fmt.Println()
	fmt.Println("## Точка минимума при epsilon:", epsilon)
	fmt.Println(" f(x):", function(u_cur))
	for _,x := range u_cur {
		fmt.Printf(" %.4f", x)
	}
	fmt.Println()
		
	// получить минимум фуркции
	return function(u_cur)
}

// Метод наискорейшего спуска(пример алгоритма)
// the method of steepest descent or stationary-phase method or saddle-point method
func SteepestDescent(bx, by, epsilon float64) float64 {
	const MAXITERATIONS = 1000
	var k int
	var x_cur float64
	var y_cur float64
	var x_next float64
	var y_next float64
	var lambda float64	
	// П.1. Задают начальное приближение и точность расчёта vec x[0], epsilon
	//	Шаг 1. Задать x[0], epsilon1 > 0, epsilon2 > 0, предельное число итераций М.
	//		Найти градиент функции в произвольной точке. Определить частные производные функции f(x):
	//		grad f(x) = [ df(x)/x1,...,df(x)/dxn ]T(транспонированный вектор)
	// Начальное приближение u[0]
	x_cur = bx
	y_cur = by

	fmt.Println("## Исходная точка:")

	fmt.Printf("  x[0]: (%f, %f)\n", x_cur, y_cur)
	
	fmt.Println()
	fmt.Println("## Приближения:")
	//Шаг 2. Положить k = 0
	stop := false
	for k = 0; k < MAXITERATIONS && !stop ; k++ {
		// П.2. Рассчитывают vec x[k+1] = vec x[k] - lambda[k] * nabla F(vec x[k]), 
		// где lambda[k]= argmin_lambda F(vec x[k] - lambda * nabla F(vec x[k]))
		//  Шаг 3. Вычислить grad f(x[k]).
		//
		// Шаг 4. Проверить выполнение критерия окончания  ||grad f(x[k])|| < epsilon1 :
		//   а) если критерий выполнен, то x[0] = x[k], останов;
		//   б) если критерий не выполнен, то перейти к шагу 5.
		//
		// Шаг 5. Проверить выполнение неравенства k ≥ M:
		//   а) если неравенство выполнено, то x[0] = x[k], останов;
		//   б) если нет, то перейти к шагу 6
		//
		// Шаг 6. Вычислить величину шага lambda[k^0] из условия
		//   F(lambda[k]) = f(x[k] - lambda[k] * grad f(x[k])) -> min lambda[k]
		//
		//
		argmin := dichotomia
		// Находим lambda[k] как минимум функции g на отрезке -10000,100000
		lambda = argmin(-10000, 100000, epsilon, x_cur, y_cur);
		// Вычисляем u[k] новое прилижение
		//  Шаг 7. Вычислить x[k+1] = x[k] - lambda[k] * grad f(x[k])
		x_next = x_cur - lambda * f_dx(x_cur, y_cur)
		y_next = y_cur - lambda * f_dy(x_cur, y_cur)
		
		fmt.Println()
		fmt.Printf("  x[%d]: (%.2f, %.2f)\n", k+1, x_next, y_next)
		fmt.Printf("  f(%.2f, %.2f) = %.2f\n", x_next, y_next, f(x_next, y_next))

		// П.3. Проверяют условие остановки:
		// Если |vec x[k+1] - vec x[k]| > epsilon
		//  или |F(vec x[k+1]) - F(vec x[k])| > epsilon
		//  или |nabla F(vec x[k+1])| > epsilon (выбирают одно из условий),
		// то k = k + 1 и переход к П.2.
		// Иначе vec x = vec x[k+1] и останов.
		//  Шаг 8. Проверить выполнение условий ||x[k+1] - x[k]|| < epsilon2, ||f(x[k+1]) - f(x[k])|| < epsilon2:
		//   а) если оба условия выполнены при текущем значении k и k = k - 1, то расчет окончен, x[0] = x[k+1], останов;
		//   б) если хотя бы одно из условий не выполнено, то положить k = k + 1 и перейти к шагу 3
		if k > 1 {
			// Проверяем условие остановки
			if norma(x_next - x_cur, y_next - y_cur) < epsilon {
				// останов
				stop = true
			}
		}
		
		x_cur = x_next
		y_cur = y_next
	}
	
	fmt.Println()
	fmt.Println("## Точка минимума epsilon:", epsilon)
	fmt.Printf("   f(%.2f , %.2f) = %.2f\n", x_cur, y_cur, f(x_cur, y_cur))
		
	// получить минимум фуркции
	return f(x_cur, y_cur)
}

